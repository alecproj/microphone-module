#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "esp_err.h"

typedef struct {
    int16_t *data;
    size_t data_size;
    int data_channels;
    int channel_index;
} audio_data;

esp_err_t downmix_to_mono(audio_data input, audio_data* output);
