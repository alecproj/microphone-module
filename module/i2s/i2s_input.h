#pragma once

#include "driver/gpio.h"
#include "esp_err.h"

// I2S interface GPIO pin assignments
#define GPIO_I2S_LRCK       (GPIO_NUM_42)   // Word select (WS)
#define GPIO_I2S_MCLK       (GPIO_NUM_NC)   // Master clock (not used)
#define GPIO_I2S_SCLK       (GPIO_NUM_41)   // Serial clock (SCK)
#define GPIO_I2S_SDIN       (GPIO_NUM_2)    // Serial data input (SDI)
#define GPIO_I2S_DOUT       (GPIO_NUM_NC)   // Serial data output (not used)

/**
 * @brief Default configuration macro for I2S standard mode
 * 
 * @param sample_rate     Audio sample rate in Hz (e.g. 16000, 44100)
 * @param channel_fmt     I2S channel slot mode (I2S_SLOT_MODE_MONO/STEREO)
 * @param bits_per_chan   Bit depth per channel (I2S_DATA_BIT_WIDTH_xxBIT)
 */
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

/**
 * @brief Initialize I2S input peripheral and driver
 * 
 * @return esp_err_t 
 *    - ESP_OK: Initialization successful
 *    - ESP_ERR_*: Initialization error
 */
esp_err_t i2s_input_init();

/**
 * @brief Get audio format descriptor string
 * 
 * @return char*    Static string indicating audio format ("MM" for stereo)
 */
char* i2s_input_get_format();

/**
 * @brief Read audio data from I2S input
 * 
 * @param buffer      Destination buffer for audio samples
 * @param buffer_len  Length of buffer in bytes
 * @return esp_err_t 
 *    - ESP_OK Data read successfully  
 *    - ESP_ERR_INVALID_ARG NULL pointer or this handle is not RX handle  
 *    - ESP_ERR_TIMEOUT Reading timeout, no reading event received from ISR within ticks_to_wait  
 *    - ESP_ERR_INVALID_STATE I2S is not ready to read  
 */
esp_err_t i2s_input_get_feed_data(int16_t *buffer, int buffer_len);
