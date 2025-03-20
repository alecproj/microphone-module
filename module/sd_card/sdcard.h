#pragma once

#include "driver/gpio.h"
#include "esp_err.h"

#define SDSPI_HOST          (SPI2_HOST)
#define GPIO_SDSPI_CS       (GPIO_NUM_2)
#define GPIO_SDSPI_SCLK     (GPIO_NUM_5)
#define GPIO_SDSPI_MISO     (GPIO_NUM_21)
#define GPIO_SDSPI_MOSI     (GPIO_NUM_18)


esp_err_t sdcard_init(char *mount_point, size_t max_files);

esp_err_t sdcard_write(const void* buffer, int size, int count, FILE* stream);
