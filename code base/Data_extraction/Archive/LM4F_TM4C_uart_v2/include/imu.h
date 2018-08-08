#ifndef IMU_H
#define IMU_H

/* Functions for reading the I2C sensors for accelerometers+gyros and accelerometers+magnetometers on the autopilot board.
 * The measurement axes are defined as follows:
 * with the board placed so that the observer is looking DOWN at the LM4F side with the 50-pin connector at the TOP, the X, Y, and Z axes are as follows:
 * X axis : vector pointing to the RIGHT
 * Y axis : vector pointing to the TOP
 * Z axis : vector pointing TOWARD the observer (UP)
 */
#define AP_VERSION MINI_AHRS
#define ADDR_ACCEL_LSM9 0x1D
#define ADDR_MAG_LSM9 0x1D
#define ADDR_GYRO_LSM9 0x6B

// #include "AP_ver.h"

#ifdef __cplusplus
extern "C" {
#endif

#define uint8_t uint8

#if AP_VERSION == 230
#define LSM303D
#define LSM330DLC
#elif AP_VERSION == 223
#define LSM303DLM
#define LSM330DL
#elif AP_VERSION == 24
#define LSM9DS0
#elif AP_VERSION == MINI_AHRS
#define LSM9DS0
#endif

#if (AP_VERSION == 223) || (AP_VERSION == 230)
#define DUAL_ACCELEROMETER
#define HAS_LSM303
#define HAS_LSM330
#endif


#if (AP_VERSION == MINI_AHRS) || (AP_VERSION == 24)
#define HAS_LSM9
#endif


enum ACCEL_UNIT
{
	UNIT_G,
	UNIT_M_SQUARED
};

enum GYRO_SCALE
{
	SCALE_250_DPS,
	SCALE_500_DPS,
	SCALE_2000_DPS
};

enum ACCELEROMETER_SCALE
{
	SCALE_2G,
	SCALE_4G,
	SCALE_8G
};

#ifdef LSM303DLM
enum MAGNETOMETER_SCALE
{
	SCALE_13GAUSS,
	SCALE_19GAUSS,
	SCALE_25GAUSS,
	SCALE_40GAUSS,
	SCALE_47GAUSS,
	SCALE_56GAUSS,
	SCALE_81GAUSS
};
#endif
#ifdef LSM303D
enum MAGNETOMETER_SCALE
{
	SCALE_2GAUSS,
	SCALE_4GAUSS,
	SCALE_8GAUSS,
	SCALE_12GAUSS
};
#endif
#ifdef LSM9DS0
enum MAGNETOMETER_SCALE
{
	SCALE_2GAUSS,
	SCALE_4GAUSS,
	SCALE_8GAUSS,
	SCALE_12GAUSS
};
#endif

typedef struct
{
	float gyro_data[3];
	float acc1_data[3];
	float acc2_data[3];
	float mag_data[3];
	enum GYRO_SCALE g;
	enum ACCELEROMETER_SCALE a;
	enum MAGNETOMETER_SCALE m;
	enum ACCEL_UNIT u;
} imu_scaled_data;

#if defined(HAS_LSM303) && defined(HAS_LSM330)
// by default, both the LSM303x chip and the LSM330x chip are enabled
// call this function (before calling the init_imu function) to disable either (or both) of these sensors chips
// supplying argument 1 disables a sensor
// this function does NOT need to be called for default behavior of enabling both sensors
void disable_imu_sensors(int disable_303, int disable_330);

// reads the disable flags for the two sensors -- 1 implies that a sensor is disabled
void get_disable_imu_sensors(int *disable_303, int *disable_330);
#endif

#ifdef HAS_LSM9
// by default, the LSM9 chip is enabled
// call this function (before calling the init_imu function) to disabl the sensor chip
// supplying argument 1 disables the sensor
// this function does NOT need to be called for default behavior of enabling the sensor
void disable_imu_sensor(int disable_imu9);

// reads the disable flag for the LSM9 sensor -- 1 implies that the sensor is disabled
void get_disable_imu_sensor(int *disable_imu9);
#endif

// initialize the imu_scaled_data struct with the specified scales/units
void init_imu_scaledStruct(imu_scaled_data* b1, enum ACCELEROMETER_SCALE a, enum MAGNETOMETER_SCALE m, enum GYRO_SCALE g, enum ACCEL_UNIT aU);

// initialize the sensor chips (skipping sensors that are disabled by calling disable_imu_sensors)
void init_imu(imu_scaled_data* b1);

// read one sample
// returns 1 if the read was successful
// returns a negative number (-a) if the read was not successful,
// ... where the least significant bit of a is 1 if the read of the LSM303x chip was unsuccessful
// ... and the second least significant bit of a is 1 if the read of the LSM330x chip was unsuccessful
int read_scaled_imu_data(imu_scaled_data* scaledBuf);

/* Example:
    imu_scaled_data imu_b1;
    init_imu_scaledStruct(&imu_b1, SCALE_4G, SCALE_4GAUSS, SCALE_250_DPS, UNIT_M_SQUARED);
    init_imu(&imu_b1);
    read_scaled_imu_data(&imu_b1);
    // now, imu_b1.gyro_data, imu_b1.acc1_data, etc., have the sensor values
    // gyro data in deg per second
    // acc data in m/s^2 if ACCEL_UNIT was specified as UNIT_M_SQUARED in call to init_imu_scaledStruct and in g otherwise
    // mag data in gauss
 */

#ifdef __cplusplus
}
#endif

#endif
