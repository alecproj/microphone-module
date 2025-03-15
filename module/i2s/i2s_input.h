#pragma once

#include "driver/gpio.h"
#include "esp_err.h"

#define GPIO_I2S_LRCK       (GPIO_NUM_11)
#define GPIO_I2S_MCLK       (GPIO_NUM_NC)
#define GPIO_I2S_SCLK       (GPIO_NUM_12)
#define GPIO_I2S_SDIN       (GPIO_NUM_10)
#define GPIO_I2S_DOUT       (GPIO_NUM_NC)

#define I2S_CONFIG_DEFAULT(sample_rate, channel_fmt, bits_per_chan) { \
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate), \
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(bits_per_chan, channel_fmt), \
        .gpio_cfg = {              \
            .mclk = GPIO_I2S_MCLK, \
            .bclk = GPIO_I2S_SCLK, \
            .ws   = GPIO_I2S_LRCK, \
            .dout = GPIO_I2S_DOUT, \
            .din  = GPIO_I2S_SDIN, \
            .invert_flags = {      \
                .mclk_inv = false, \
                .bclk_inv = false, \
                .ws_inv = false,   \
            },                     \
        },                         \
    }


esp_err_t i2s_input_init();

char* i2s_input_get_format();

esp_err_t i2s_input_get_feed_data(int16_t *buffer, int buffer_len);
