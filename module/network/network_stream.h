#pragma once

#include "esp_err.h"
#include "lwip/err.h"
#include "lwip/sockets.h"

// Network and RTP configuration constants
#define MAX_FRAGMENT_SIZE 1024        // Maximum audio fragment size in bytes
#define RTP_HEADER_SIZE 12            // Size of RTP header in bytes
#define MAX_PACKET_SIZE (MAX_FRAGMENT_SIZE + RTP_HEADER_SIZE)  // Max UDP packet size

#define SERVER_IP "192.168.0.175"     // Default server IP address
#define SERVER_PORT 8765              // Default server UDP port

/**
 * @brief RTP packet header metadata
 */
typedef struct {
    uint32_t audio_ssrc;
    uint16_t sequence_number; 
    uint32_t time_stamp;
} rtp_data_t;

/**
 * @brief Network connection context
 */
typedef struct {
    int udp_sock;
    struct sockaddr_in server_addr;
    socklen_t server_addrlen;
    rtp_data_t rtp_data;
} network_data_t;

/**
 * @brief Macro to initialize RTP header byte array
 * 
 * @param rtp_data RTP metadata structure
 */
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

/**
 * @brief Initialize network connection and RTP metadata
 * 
 * @param network_data Network context structure
 * @return esp_err_t 
 *    - ESP_OK: Initialization successful
 *    - ESP_ERR_INVALID_ARG: NULL pointer to network_data
 *    - ESP_FAIL: Socket creation error
 */
esp_err_t network_init(network_data_t *network_data);

/**
 * @brief Reinitialize UDP socket connection
 * 
 * @param sock Pointer to socket descriptor
 * @return esp_err_t 
 *    - ESP_OK: Reinitialization successful
 *    - ESP_ERR_INVALID_ARG: Invalid socket pointer
 *    - ESP_FAIL: Socket recreation error
 */
esp_err_t udp_socket_reinit(int *sock);

/**
 * @brief Construct RTP header from metadata (Deprecated)
 * 
 * @param rtp_data RTP metadata structure
 * @param packet Output buffer for header bytes
 */
void udp_prepare_rtp_header(rtp_data_t *rtp_data, uint8_t *packet);

/**
 * @brief Send PCM audio data over UDP with RTP encapsulation
 * 
 * @param data Pointer to PCM audio samples
 * @param data_size Size of audio data in bytes
 * @param network_data Initialized network context
 */
void udp_send_pcm(int16_t *data, size_t data_size, network_data_t *network_data);
