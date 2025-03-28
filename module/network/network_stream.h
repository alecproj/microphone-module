#pragma once

#include "esp_err.h"
#include "lwip/err.h"
#include "lwip/sockets.h"

#define MAX_FRAGMENT_SIZE 1024 
#define RTP_HEADER_SIZE 12
#define MAX_PACKET_SIZE (MAX_FRAGMENT_SIZE + RTP_HEADER_SIZE)

#define SERVER_IP "192.168.0.175"
#define SERVER_PORT 8765

typedef struct {
    uint32_t audio_ssrc;
    uint16_t sequence_number; 
    uint32_t time_stamp;
} rtp_data_t;

typedef struct {
    int udp_sock;
    struct sockaddr_in server_addr;
    socklen_t server_addrlen;
    rtp_data_t rtp_data;
} network_data_t;

#define RTP_HEADER_INIT(rtp_data) {             \
    /* Version 2 */                             \
    0x80,                                       \
    /* Payload Type 10 (L16 PCM mono) */        \
    0x0A,                                       \
    /* Sequence Number (16 bit) */              \
    (rtp_data.sequence_number >> 8) & 0xFF,     \
    rtp_data.sequence_number & 0xFF,            \
    /* Timestamp (32 bit) */                    \
    (rtp_data.time_stamp >> 24) & 0xFF,         \
    (rtp_data.time_stamp >> 16) & 0xFF,         \
    (rtp_data.time_stamp >> 8) & 0xFF,          \
    rtp_data.time_stamp & 0xFF,                 \
    /* SSRC (32 bit) */                         \
    (rtp_data.audio_ssrc >> 24) & 0xFF,         \
    (rtp_data.audio_ssrc >> 16) & 0xFF,         \
    (rtp_data.audio_ssrc >> 8) & 0xFF,          \
    rtp_data.audio_ssrc & 0xFF }

esp_err_t network_init(network_data_t *network_data);

esp_err_t udp_socket_reinit(int *sock);

void udp_prepare_rtp_header(rtp_data_t *rtp_data, uint8_t *packet);

void udp_send_pcm(int16_t *data, size_t data_size, network_data_t *network_data);
