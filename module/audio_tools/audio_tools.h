#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "esp_err.h"

/**
 * @brief The chunk of audio data with metadata for processing
 */
typedef struct {
    int16_t *data;      // pointer to raw audio samples (e.g., PCM).
    size_t data_size;   // size of the data buffer in bytes.
    int data_channels;  // number of audio channels (e.g., 1 for mono, 2 for stereo)
    int target_channel;  // channel index to prioritize during downmix (e.g., 0 for first channel)
} audio_data_t;

/**
 * @brief Function for mixing multichannel audio data into mono.
 *
 * @param input     Multichannel audio data
 * @param output    The result of function - mono audio data
 * @return
 *    - ESP_OK: Success
 *    - Others: Fail
 */
esp_err_t downmix_to_mono(audio_data_t input, audio_data_t* output);

