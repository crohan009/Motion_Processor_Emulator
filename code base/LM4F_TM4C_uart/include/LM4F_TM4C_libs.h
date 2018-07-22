#ifndef LM4F_TM4C_LIBS_H
#define LM4F_TM4C_LIBS_H

// init communications with the RM48
void init_AP_comms();

// send a string message to the RM48
void AP_send_msg(const char *s);


// struct definitions
#include "pxflow_struct_defn.h"
#include "positioning_update_struct_defn.h"


// send a PXFlow struct message to the RM48
void AP_send_pxflow_msg(mavlink_optical_flow_t *mavlink_msg);

// send a positioning_update struct message to the RM48
void AP_send_positioning_update_msg(positioning_update_t *positioning_update_msg);

// send a generic message of n floats to the RM48
void AP_send_n_floats_msg(float *f, int n);


#endif
