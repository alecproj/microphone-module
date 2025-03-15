#include "i2s_input.h"

#include "freertos/FreeRTOS.h"
#include "driver/i2s_std.h"


static i2s_chan_handle_t rx_handle = NULL;

esp_err_t i2s_input_init()
{
    esp_err_t rv = ESP_OK;
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    i2s_std_config_t std_cfg = I2S_CONFIG_DEFAULT(16000, I2S_SLOT_MODE_STEREO, I2S_DATA_BIT_WIDTH_32BIT);

    rv |= i2s_new_channel(&chan_cfg, NULL, &rx_handle);
    rv |= i2s_channel_init_std_mode(rx_handle, &std_cfg);
    rv |= i2s_channel_enable(rx_handle);

    return rv;
}

char* i2s_input_get_format()
{
    return "MM";
}

esp_err_t i2s_input_get_feed_data(int16_t *buffer, int buffer_len)
{
    esp_err_t rv = ESP_OK;
    size_t bytes_read;
    int audio_chunksize = buffer_len / (sizeof(int32_t) * 2);
    int32_t *tmp_buff = NULL;

    rv = i2s_channel_read(rx_handle, buffer, buffer_len, &bytes_read, portMAX_DELAY);

    tmp_buff = buffer;
    for (int i = 0; i < audio_chunksize; i++) {
        tmp_buff[2 * i] = tmp_buff[2 * i] >> 14;
        tmp_buff[2 * i + 1] = tmp_buff[2 * i + 1] >> 14;
    }

    return rv;
}
