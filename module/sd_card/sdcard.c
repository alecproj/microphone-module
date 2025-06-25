#include "sdcard.h"

#include <unistd.h>

#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"


static sdmmc_card_t *card;
static const char *TAG = "sdcard";

esp_err_t sdcard_init(char *mount_point, size_t max_files)
{
    if (NULL != card)
    {
        return ESP_ERR_INVALID_STATE;
    }
    esp_err_t rv = ESP_OK;

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = max_files,
        .allocation_unit_size = 16 * 1024
    };
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    spi_bus_config_t bus_cfg = {
        .mosi_io_num = GPIO_SDSPI_MOSI,
        .miso_io_num = GPIO_SDSPI_MISO,
        .sclk_io_num = GPIO_SDSPI_SCLK,
        .quadwp_io_num = GPIO_NUM_NC,
        .quadhd_io_num = GPIO_NUM_NC,
        .max_transfer_sz = 4000,
    };
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = GPIO_SDSPI_CS;
    slot_config.host_id = host.slot;

    rv = spi_bus_initialize(host.slot, &bus_cfg, SPI_DMA_CH_AUTO);
    if (rv != ESP_OK) 
    {
        ESP_LOGE(TAG, "Failed to initialize bus.");
        return rv;
    }

    esp_vfs_fat_sdspi_mount(mount_point, &host, &slot_config, &mount_config, &card);

    /* Check for mount result. */
    if (rv != ESP_OK) 
    {
        if (rv == ESP_FAIL) 
        {
            ESP_LOGE(TAG, "Failed to mount filesystem. "
                     "If you want the card to be formatted," 
                     " set the EXAMPLE_FORMAT_IF_MOUNT_FAILED"
                     " menuconfig option.");
        } else 
        {
            ESP_LOGE(TAG, "Failed to initialize the card (%s). "
                     "Make sure SD card lines have pull-up "
                     "resistors in place.", esp_err_to_name(rv));
        }
        return rv;
    }


    /* Card has been initialized, print its properties. */
    sdmmc_card_print_info(stdout, card);

    return rv;
}

esp_err_t sdcard_write(const void* buffer, int size, int count, FILE* stream)
{
    esp_err_t res = ESP_OK;
    res = fwrite(buffer, size, count, stream);
    res |= fflush(stream);
    res |= fsync(fileno(stream));

    return res;
}
