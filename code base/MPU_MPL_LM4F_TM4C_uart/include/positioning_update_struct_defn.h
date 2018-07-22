#ifndef POSITIONING_UPDATE_STRUCT_DEFN_H
#define POSITIONING_UPDATE_STRUCT_DEFN_H

typedef struct __positioning_update_t
{
	float x_vel;
	float y_vel;
	float x_optical_flow;
	float y_optical_flow;
	float sonar_height_measurement;
	float lidar_height_measurement;
	uint64_t flow_data_timestamp;
} positioning_update_t;


#endif
