#ifndef PXFLOW_STRUCT_DEFN_H
#define PXFLOW_STRUCT_DEFN_H

#include <stdint.h>

typedef struct __mavlink_optical_flow_t
{
 uint64_t time_usec; ///< Timestamp (UNIX)
 float flow_comp_m_x; ///< Flow in meters in x-sensor direction, angular-speed compensated
 float flow_comp_m_y; ///< Flow in meters in y-sensor direction, angular-speed compensated
 float ground_distance; ///< Ground distance in meters. Positive value: distance known. Negative value: Unknown distance
 int16_t flow_x; ///< Flow in pixels * 10 in x-sensor direction (dezi-pixels)
 int16_t flow_y; ///< Flow in pixels * 10 in y-sensor direction (dezi-pixels)
 uint8_t sensor_id; ///< Sensor ID
 uint8_t quality; ///< Optical flow quality / confidence. 0: bad, 255: maximum quality
} mavlink_optical_flow_t;


#endif
