#pragma once

#include "driver/gpio.h"
#include "esp_err.h"

// SD Card SPI interface configuration
#define SDSPI_HOST          (SPI2_HOST)    // SPI host peripheral (SPI2)
#define GPIO_SDSPI_CS       (GPIO_NUM_13)  // Chip Select (CS) pin
#define GPIO_SDSPI_SCLK     (GPIO_NUM_12)  // Serial Clock (SCK) pin
#define GPIO_SDSPI_MISO     (GPIO_NUM_10)  // Master Input Slave Output (MISO) pin
#define GPIO_SDSPI_MOSI     (GPIO_NUM_11)  // Master Output Slave Input (MOSI) pin

/**
 * @brief Initialize and mount SD card using SPI interface
 * 
 * @param mount_point  Filesystem mount point (e.g., "/sdcard")
 * @param max_files    Maximum number of open files in FATFS
 * @return esp_err_t 
 *    - ESP_OK: SD card mounted successfully
 *    - ESP_ERR_INVALID_STATE: Already initialized
 *    - ESP_ERR_NO_MEM: Memory allocation error
 *    - ESP_FAIL: Mount failed (check card and wiring)
 */
esp_err_t sdcard_init(char *mount_point, size_t max_files);

/**
 * @brief Write data to SD card with full synchronization
 * 
 * @param buffer  Pointer to data buffer
 * @param size    Size of each data element (bytes)
 * @param count   Number of elements to write
 * @param stream  File stream pointer (from fopen)
 * @return esp_err_t 
 *    - ESP_OK: Write and sync successful
 *    - Other: Error code from file operations
 */
esp_err_t sdcard_write(const void* buffer, int size, int count, FILE* stream);
