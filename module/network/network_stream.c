#include "network_stream.h"

#include <string.h>

#include "esp_log.h"
#include "freertos/task.h"


static const char *TAG = "network";

esp_err_t network_init(network_data_t *network_data)
{
    if (network_data == NULL) return ESP_ERR_INVALID_ARG;
    network_data->udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (network_data->udp_sock == -1)
    {
        ESP_LOGE(TAG, "socket: %s", strerror(errno));
        return ESP_FAIL;
    }
    network_data->server_addr.sin_family = AF_INET;
    network_data->server_addr.sin_port = htons(SERVER_PORT);
    network_data->server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    network_data->server_addrlen = sizeof(network_data->server_addr);

    network_data->rtp_data.audio_ssrc = 0x12345678; 
    network_data->rtp_data.sequence_number = 0;
    network_data->rtp_data.time_stamp = 0;

    return ESP_OK;
}

esp_err_t udp_socket_reinit(int *sock)
{
    if (!sock) return ESP_ERR_INVALID_ARG;
    if (*sock != -1)
    {
        ESP_LOGD(TAG, "Shutting down socket and restarting...");
        shutdown(*sock, 0);
        close(*sock);
    }

    *sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (*sock == -1)
    {
        ESP_LOGE(TAG, "socket: %s", strerror(errno));
        return ESP_FAIL;
    }
    return ESP_OK;
}

// TODO: Remove
void udp_prepare_rtp_header(rtp_data_t *rtp_data, uint8_t *packet)
{
    /* Version 2 */
    packet[0] = 0x80;
    /* Payload Type 10 (L16 PCM mono) */
    packet[1] = 0x0A;
    /* Sequence Number (16 bit) */
    packet[2] = (rtp_data->sequence_number >> 8) & 0xFF;
    packet[3] = rtp_data->sequence_number & 0xFF;
    /* Timestamp (32 bit) */
    packet[4] = (rtp_data->time_stamp >> 24) & 0xFF;
    packet[5] = (rtp_data->time_stamp >> 16) & 0xFF;
    packet[6] = (rtp_data->time_stamp >> 8) & 0xFF;
    packet[7] = rtp_data->time_stamp & 0xFF;
    /* SSRC (32 bit) */
    packet[8] = (rtp_data->audio_ssrc >> 24) & 0xFF;
    packet[9] = (rtp_data->audio_ssrc >> 16) & 0xFF;
    packet[10] = (rtp_data->audio_ssrc >> 8) & 0xFF;
    packet[11] = rtp_data->audio_ssrc & 0xFF;
}

void udp_send_pcm(int16_t *data, size_t data_size, network_data_t *network_data) 
{
    if (!data || !network_data)
    {
        ESP_LOGE(TAG, "udp_send_pcm: Invalid pointer passed");
    }
    size_t fragment_offset = 0;
    while (fragment_offset < data_size) 
    {
        int fragment_size = MAX_FRAGMENT_SIZE;
        if (fragment_size + fragment_offset > data_size) 
        {
            fragment_size = data_size - fragment_offset;
        }
        int packet_size = fragment_size + RTP_HEADER_SIZE;
        uint8_t packet[MAX_PACKET_SIZE] = RTP_HEADER_INIT(network_data->rtp_data);

        /* Copy audio data to package (little-endian) */
        memcpy(packet + RTP_HEADER_SIZE, data, fragment_size);

        /* Send packet */
        ssize_t sended_size = sendto(
                network_data->udp_sock, packet, packet_size, 0, 
                (struct sockaddr*)&network_data->server_addr,
                network_data->server_addrlen);
        if (sended_size != packet_size)
        {
            ESP_LOGE(TAG, "sendto: %s", strerror(errno));
            if (network_data->udp_sock < 0)
            {
                udp_socket_reinit(&network_data->udp_sock);
            }
            return;
        }

        fragment_offset += fragment_size;
        network_data->rtp_data.sequence_number++;
        network_data->rtp_data.time_stamp += fragment_size / 2; // number of samples
    }
}
