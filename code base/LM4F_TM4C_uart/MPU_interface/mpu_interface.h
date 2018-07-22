#ifndef MPU_INTERFACE_H_
#define MPU_INTERFACE_H_

typedef struct _MPU_MPL_outputs
{
  double quat[4];
  float acc[3]; // includes gravity
  float gyr[3];
  float mag[3];
  signed char quat_accuracy;
  signed char acc_accuracy;
  signed char gyr_accuracy;
  signed char mag_accuracy;
  signed long quat_timestamp;
  signed long acc_timestamp;
  signed long gyr_timestamp;
  signed long mag_timestamp;
} MPU_MPL_outputs;

typedef struct
{
    float angles[3];
    signed char accuracy;
    unsigned long timestamp;
} EulerAngles_outputs;

typedef struct
{
    float R[9];
    signed char accuracy;
    unsigned long timestamp;
} RotationMatrix_outputs;

// fifo_rate in Hz, mpl_rate in number of fifo samples
int mpu_open(unsigned int fifo_rate, unsigned int mpl_rate);
// returns 1 if updated data is returned
int mpu_update(MPU_MPL_outputs *mpu_mpl_outputs);
void mpu_get_outputs(MPU_MPL_outputs *mpu_mpl_outputs);
void mpu_get_euler(EulerAngles_outputs *angles);
void mpu_get_rot_mat(RotationMatrix_outputs *R);

int test_mpu();

#endif /* MPU_INTERFACE_H_ */
