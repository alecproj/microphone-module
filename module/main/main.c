#include <stdbool.h>

#include "esp_afe_sr_models.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "model_path.h"

#include "i2s_input.h"
#include "sdcard.h"
#include "audio_tools.h"


static esp_afe_sr_iface_t *afe_handle = NULL;
static bool sdcard_enable = true;

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

void fetch_Task(void *arg)
{
    esp_afe_sr_data_t *afe_data = arg;
    FILE* fd = NULL;

    if (sdcard_enable) 
    {
        fd = fopen("/sdcard/TEST.pcm", "w+");
        if(fd == NULL) 
        {
            printf("can not open file!\n");
        }
    }

    while (1)
    {
        afe_fetch_result_t *res = afe_handle->fetch(afe_data);
        if (!res || res->ret_value == ESP_FAIL)
        {
            printf("fetch error\n");
            break;
        }
        /* printf("vad state: %s\n", res->vad_state == VAD_SILENCE ? "noise" : "speech"); */
        if (res->vad_state == VAD_SPEECH)
        {
            printf("vad state: speech\n");
        }

        if (sdcard_enable) 
        {
            /* Save speech data */
            if (res->vad_cache_size > 0) 
            {
                esp_err_t rv = ESP_OK;
                audio_data_t input = {
                    .data = res->vad_cache,
                    .data_size = res->vad_cache_size,
                    .data_channels = afe_handle->get_feed_channel_num(afe_data),
                    .target_channel = res->trigger_channel_id
                };
                audio_data_t output = {0};

                rv = downmix_to_mono(input, &output);
                if (rv == ESP_OK)
                {
                    printf("Save vad cache: %d\n", res->vad_cache_size);
                    sdcard_write(output.data, 1, output.data_size, fd);
                }
                else printf("ERROR stereo_to_mono");

                free(output.data);
                output.data = NULL;
            }
            if (res->vad_state == VAD_SPEECH) 
            {
                esp_err_t rv = ESP_OK;
                audio_data_t input = {
                    .data = res->data,
                    .data_size = res->data_size,
                    .data_channels = afe_handle->get_feed_channel_num(afe_data),
                    .target_channel = res->trigger_channel_id
                };
                audio_data_t output = {0};

                rv = downmix_to_mono(input, &output);
                if (rv == ESP_OK)
                {
                    sdcard_write(output.data, 1, output.data_size, fd);
                }
                else printf("ERROR stereo_to_mono");

                free(output.data);
                output.data = NULL;
            }
        }
    }

    vTaskDelete(NULL);
}

void app_main()
{
    ESP_ERROR_CHECK(i2s_input_init());
    if (sdcard_enable)
    {
        ESP_ERROR_CHECK(sdcard_init("/sdcard", 10));
    }

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
