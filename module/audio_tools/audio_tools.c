#include "audio_tools.h"

esp_err_t downmix_to_mono(audio_data_t input, audio_data_t* output)
{
    if (!input.data || !output || input.data_channels <= 0 ||
        input.target_channel < 0 || input.target_channel >= input.data_channels) 
    {
        return ESP_ERR_INVALID_ARG;
    }

    int total_samples = input.data_size / sizeof(int16_t);  
    int samples_per_channel = total_samples / input.data_channels;
    if (samples_per_channel <= 0) 
    {
        return ESP_ERR_INVALID_SIZE;
    }

    output->data_size = samples_per_channel * sizeof(int16_t);
    output->data = malloc(output->data_size);
    assert(output->data);

    for (int i = 0; i < samples_per_channel; i++) 
    {
        output->data[i] = input.data[i * input.data_channels + input.target_channel];
    }

    output->target_channel = 0;
    output->data_channels = 1;

    return ESP_OK;
}
