#include <stdbool.h>

#include "esp_afe_sr_models.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "model_path.h"
#include "esp_log.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "protocol_examples_common.h"

#include "i2s_input.h"
#include "sdcard.h"
#include "audio_tools.h"
#include "network_stream.h"

// Configuration switches
#define SD_FOR_DEBUG_EN 0  // Enable SD card logging for debug (0=disabled)
#define NETWORK_EN 1       // Enable network streaming (1=enabled)

static esp_afe_sr_iface_t *afe_handle = NULL;
static const char* TAG = "module";
#if SD_FOR_DEBUG_EN
static FILE* fd = NULL;
#endif // SD_FOR_DEBUG_EN
#if NETWORK_EN
static network_data_t network_data = {0};
#endif // NETWORK_EN

/**
 * @brief Process audio data and stream to output targets
 * 
 * @param audio_data Input audio data structure
 * 
 * Performs:
 * 1. Downmix to mono (if multichannel input)
 * 2. SD card debug logging (if enabled)
 * 3. Network streaming (if enabled)
 * 4. Memory cleanup
 */
void postprocess_and_stream(audio_data_t* audio_data)
{
    esp_err_t rv = ESP_OK;
    audio_data_t output = {0};

    if (!audio_data)
    {
        ESP_LOGE(TAG, "An invalid pointer was passed to the "
                      "post processing function.");
        return;
    }

    if (audio_data->data_channels > 1)
    {
        rv = downmix_to_mono(*audio_data, &output);
        if (rv != ESP_OK)
        {
            ESP_LOGE(TAG, "An error (%s) occurred during post processing.", 
                     esp_err_to_name(rv));
            return;
        }
    }

#if SD_FOR_DEBUG_EN
    sdcard_write(output.data, 1, output.data_size, fd);
#endif // SD_FOR_DEBUG_EN

#if NETWORK_EN
    udp_send_pcm(output.data, output.data_size, &network_data);
#endif // NETWORK_EN

    free(output.data);
    output.data = NULL;
}

/**
 * @brief Task for feeding audio data to AFE (Audio Front-End)
 * 
 * @param arg AFE data handle (esp_afe_sr_data_t)
 * 
 * Continuously:
 * 1. Reads audio from I2S input
 * 2. Feeds data to AFE processing pipeline
 */
void feed_Task(void *arg)
{
    esp_afe_sr_data_t *afe_data = arg;
    int feed_chunksize = afe_handle->get_feed_chunksize(afe_data);
    // TODO: configuration compliance check
    int feed_nch = afe_handle->get_feed_channel_num(afe_data);
    int16_t *feed_buff = (int16_t *) malloc(feed_chunksize * feed_nch * sizeof(int16_t));
    assert(feed_buff);

    while (1)
    {
        i2s_input_get_feed_data(
            feed_buff, 
            feed_chunksize * feed_nch * sizeof(int16_t)
        );
        afe_handle->feed(afe_data, feed_buff);
    }
    
    if (feed_buff)
    {
        free(feed_buff);
        feed_buff = NULL;
    }
    vTaskDelete(NULL);
}

/**
 * @brief Task for fetching processed audio from AFE
 * 
 * @param arg AFE data handle (esp_afe_sr_data_t)
 * 
 * Continuously:
 * 1. Fetches processed audio from AFE
 * 2. Processes VAD (Voice Activity Detection) results
 * 3. Sends valid audio segments to postprocessing
 */
void fetch_Task(void *arg)
{
    esp_afe_sr_data_t *afe_data = arg;

    while (1)
    {
        // Get processed audio results
        afe_fetch_result_t *res = afe_handle->fetch(afe_data);
        if (!res || res->ret_value == ESP_FAIL)
        {
            ESP_LOGE(TAG, "Fetch data error.");
            break;
        }

        if (res->vad_cache_size > 0)
        {
            ESP_LOGI(TAG, "VAD cache size: %d", res->vad_cache_size);
            audio_data_t audio_data = {
                .data = res->vad_cache,
                .data_size = res->vad_cache_size,
                .data_channels = afe_handle->get_feed_channel_num(afe_data),
                .target_channel = res->trigger_channel_id
            };
            postprocess_and_stream(&audio_data);
        }
        if (res->vad_state == VAD_SPEECH)
        {
            ESP_LOGI(TAG, "VAD state: speech");
            audio_data_t audio_data = {
                .data = res->data,
                .data_size = res->data_size,
                .data_channels = afe_handle->get_feed_channel_num(afe_data),
                .target_channel = res->trigger_channel_id
            };
            postprocess_and_stream(&audio_data);
        }
    }

    vTaskDelete(NULL);
}

/**
 * @brief Main application entry point
 * 
 * Initializes:
 * 1. I2S audio input
 * 2. Network connectivity (if enabled)
 * 3. SD card (if debug enabled)
 * 4. Audio Front-End processing
 * 5. Feed/fetch tasks
 */
void app_main()
{
    ESP_ERROR_CHECK(i2s_input_init());
#if NETWORK_EN
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
     * TODO: Replace
     */
    ESP_ERROR_CHECK(example_connect());
    ESP_ERROR_CHECK(network_init(&network_data));
#endif // NETWORK_EN

#if SD_FOR_DEBUG_EN
    ESP_ERROR_CHECK(sdcard_init("/sdcard", 10));
    fd = fopen("/sdcard/TEST.pcm", "w+");
    if(fd == NULL) 
    {
        ESP_LOGE(TAG, "Error opening file.");
    }
#endif // SD_FOR_DEBUG_EN

    /* Initialize AFE Configuration */
    srmodel_list_t *models = esp_srmodel_init("model");
    afe_config_t *afe_config = afe_config_init(i2s_input_get_format(), models, AFE_TYPE_VC, AFE_MODE_HIGH_PERF);
    afe_config->vad_min_noise_ms = 1000;
    afe_config->vad_min_speech_ms = 128;
    afe_config->vad_mode = VAD_MODE_0;
    afe_config->wakenet_init = 0;

    /* Create AFE Instance */
    afe_handle = esp_afe_handle_from_config(afe_config);
    esp_afe_sr_data_t *afe_data = afe_handle->create_from_config(afe_config);
    afe_config_free(afe_config);

    /* Create Threads */
    xTaskCreatePinnedToCore(&feed_Task, "feed", 8 * 1024, (void*)afe_data, 5, NULL, 0);
    xTaskCreatePinnedToCore(&fetch_Task, "fetch", 4 * 1024, (void*)afe_data, 5, NULL, 1);
}
