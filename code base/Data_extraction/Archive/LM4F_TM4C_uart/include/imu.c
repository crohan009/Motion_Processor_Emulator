#include "imu.h"
#include "i2c_tm4c.h"
#include <stdio.h>
#include <stdint.h>

// void AP_send_msg(const char *s);


/*
imu.c -> External Peripheral library for the LSM303DLM combo accelerometer + magnetometer
                sensor from STmicroelectronics. Currently using polling to set up and
                operate both sensors at an I2C bus speed of 100 KHz by default.
(Note: Bus speed can be changed to 400 KHz by commenting out the MODE_100 statement and
        commenting in the MODE_400 define statement)

(c) 2012-2013,  Abhimanyu Ghosh, Controls and Robotics Research Lab (CRRL)
                Polytechnic Institute of NYU
*/
int *f;

#ifdef HAS_LSM303
enum LSM303_ACCEL_SCALE
{
	LSM303_SCALE_2_G,
	LSM303_SCALE_4_G,
	LSM303_SCALE_8_G
};
#endif

#ifdef HAS_LSM330
enum LSM330_ACCEL_SCALE
{
	LSM330_SCALE_2_G,
	LSM330_SCALE_4_G,
	LSM330_SCALE_8_G,
	LSM330_SCALE_16_G
};
#endif

#ifdef HAS_LSM9
enum LSM9_ACCEL_SCALE
{
	LSM9_SCALE_2_G,
	LSM9_SCALE_4_G,
	LSM9_SCALE_6_G,
	LSM9_SCALE_8_G,
	LSM9_SCALE_16_G,
};
#endif

//#define uint32_t uint32
#define AXIS_X 0
#define AXIS_Y 1
#define AXIS_Z 2

#define bit(n) (1<<(n))
int to_signed_int_12(uint8_t low_byte, uint8_t high_byte); // Assumed left-aligned data
int to_signed_int_16(uint8_t low_byte, uint8_t high_byte);

#ifdef LSM303DLM
#define LSM303_ACCEL_SCALE_FACTOR_2G ( 0.0010f / 16.0f)
#define LSM303_ACCEL_SCALE_FACTOR_4G ( 0.0020f / 16.0f)
#define LSM303_ACCEL_SCALE_FACTOR_8G ( 0.0039f / 16.0f)
#endif
#ifdef LSM303D
#define LSM303_ACCEL_SCALE_FACTOR_2G (6.1e-5)
#define LSM303_ACCEL_SCALE_FACTOR_4G (1.22e-4)
#define LSM303_ACCEL_SCALE_FACTOR_8G (2.44e-4)
#endif

#ifdef HAS_LSM330
#define LSM330_ACCEL_SCALE_FACTOR_2G  (0.0010f / 16.0f)
#define LSM330_ACCEL_SCALE_FACTOR_4G  (0.0020f / 16.0f)
#define LSM330_ACCEL_SCALE_FACTOR_8G  (0.0040f / 16.0f)
#define LSM330_ACCEL_SCALE_FACTOR_16G (0.0120f / 16.0f)
#endif

#ifdef LSM9DS0
#define LSM9_ACCEL_SCALE_FACTOR_2G (6.1e-5)
#define LSM9_ACCEL_SCALE_FACTOR_4G (1.22e-4)
#define LSM9_ACCEL_SCALE_FACTOR_6G (1.83e-4)
#define LSM9_ACCEL_SCALE_FACTOR_8G (2.44e-4)
#define LSM9_ACCEL_SCALE_FACTOR_16G (7.32e-4)
#endif

#ifdef LSM303DLM
#define K_MAGNETOMETER_13_xy 0.000909091f
#define K_MAGNETOMETER_13_z  0.001020408f
#define K_MAGNETOMETER_19_xy 0.001169591f
#define K_MAGNETOMETER_19_z  0.001315789f
#define K_MAGNETOMETER_25_xy 0.001492537f
#define K_MAGNETOMETER_25_z  0.001666667f
#define K_MAGNETOMETER_40_xy 0.002222222f
#define K_MAGNETOMETER_40_z  0.002500000f
#define K_MAGNETOMETER_47_xy 0.002500000f
#define K_MAGNETOMETER_47_z  0.002816901f
#define K_MAGNETOMETER_56_xy 0.003030303f
#define K_MAGNETOMETER_56_z  0.003389831f
#define K_MAGNETOMETER_81_xy 0.004347826f
#define K_MAGNETOMETER_81_z  0.004878049f
#endif
#ifdef LSM303D
#define K_MAGNETOMETER_2_xy  8.0e-5
#define K_MAGNETOMETER_2_z   8.0e-5
#define K_MAGNETOMETER_4_xy  1.6e-4
#define K_MAGNETOMETER_4_z   1.6e-4
#define K_MAGNETOMETER_8_xy  3.2e-4
#define K_MAGNETOMETER_8_z   3.2e-4
#define K_MAGNETOMETER_12_xy 4.79e-4
#define K_MAGNETOMETER_12_z  4.79e-4
#endif
#ifdef LSM9DS0
#define K_MAGNETOMETER_2_xy  8.0e-5
#define K_MAGNETOMETER_2_z   8.0e-5
#define K_MAGNETOMETER_4_xy  1.6e-4
#define K_MAGNETOMETER_4_z   1.6e-4
#define K_MAGNETOMETER_8_xy  3.2e-4
#define K_MAGNETOMETER_8_z   3.2e-4
#define K_MAGNETOMETER_12_xy 4.8e-4
#define K_MAGNETOMETER_12_z  4.8e-4
#endif

#define K_GYRO_250 	(0.00875f )
#define K_GYRO_500 	(0.01750f )
#define K_GYRO_2000 (0.07000f )

short int twosComp_16b(uint8_t low_byte, uint8_t high_byte);
void twosComp_sensorData_16b(uint8_t data[6], int output[3]);

typedef struct
{
	int gyro_data[3];
	int acc1_data[3];
#ifdef DUAL_ACCELEROMETER
	int acc2_data[3];
#endif
	int mag_data[3];
}imu_raw_data;
int read_raw_imu_data(imu_raw_data* dataBuf);

#ifdef HAS_LSM303
int accel_lsm303_write_reg(uint8_t reg, uint8_t val);
int lsm303_accel_read_all_data(uint8_t ret[6]);
int accel_lsm303_setup(void);
int accel_lsm303_set_scale(enum LSM303_ACCEL_SCALE scale);
#endif
#ifdef HAS_LSM330
int accel_lsm330_write_reg(uint8_t reg, uint8_t val);
int lsm330_accel_read_all_data(uint8_t ret[6]);
int accel_lsm330_setup(void);
int accel_lsm330_set_scale(enum LSM330_ACCEL_SCALE scale);
#endif
#ifdef HAS_LSM9
int accel_lsm9_write_reg(uint8_t reg, uint8_t val);
int lsm9_accel_read_all_data(uint8_t ret[6]);
int accel_lsm9_setup(void);
int accel_lsm9_set_scale(enum LSM9_ACCEL_SCALE scale);
#endif

int mag_write_reg(uint8_t reg, uint8_t val);
int gyro_write_reg(uint8_t reg, uint8_t val);
int mag_read_all_data(uint8_t ret[6]);
int gyro_read_all_data(uint8_t ret[6]);
int gyro_setup(void);
int gyro_set_scale(enum GYRO_SCALE g);
int mag_setup(void);
int mag_set_scale(enum MAGNETOMETER_SCALE m);


#if defined(HAS_LSM303) && defined(HAS_LSM330)
static int disable_303 = 0;
static int disable_330 = 0;
static int *disable_accel1;
static int *disable_accel2;
#endif
#ifdef HAS_LSM9
static int disable_imu9 = 0;
static int *disable_accel;
#endif
static int *disable_mag;
static int *disable_gyr;

//static void delay_ms(int ms)
//{
//	int i = 0;
//	int j = ms*46250;
//	for(i = 0; i < j; ++i)
//	{
//		++i;
//	}
//}
//void delay_ms (unsigned  long time  )
//{
//    SysCtlDelay(time * (SysCtlClockGet() / 3 / 1000));
//}
#if defined(HAS_LSM303) && defined(HAS_LSM330)
void disable_imu_sensors(int _disable_303, int _disable_330)
{
	disable_303 = _disable_303;
	disable_330 = _disable_330;
}

void get_disable_imu_sensors(int *_disable_303, int *_disable_330)
{
	*_disable_303 = disable_303;
	*_disable_330 = disable_330;
}
#endif

#ifdef HAS_LSM9
void disable_imu_sensor(int _disable_imu9)
{
	disable_imu9 = _disable_imu9;
}

void get_disable_imu_sensor(int *_disable_imu9)
{
	*_disable_imu9 = disable_imu9;
}
#endif


/*
 * Initializes LSM330 and LSM303 peripherals (2x accelerometers, gyro, magnetometer)
 * Arguments: GYRO_SCALE g: Scale factor for gyro (Can be of value SCALE_250_DPS, SCALE_500_DPS, SCALE_2000_DPS)
 * 				ACCELEROMETER_SCALE a: Scale factor for accelerometer (Can be of value SCALE_2G, SCALE_4G, SCALE_8G)
 * 				MAGNETOMETER_SCALE m: Scale factor for magnetometer (Please look at imu.h, or mouse over for possible values)
 * 				imu_scaled_data* b1: Pointer to scaled IMU data structure to store scale factor settings and IMU scaled
 * 										data.
 */

static void make_disable_ptrs()
{
#if defined(HAS_LSM303) && defined(HAS_LSM330)
	disable_accel1 = &disable_330;
	disable_accel2 = &disable_303;
	disable_mag = &disable_303;
	disable_gyr = &disable_330;
#endif
#if defined(HAS_LSM9)
	disable_accel = &disable_imu9;
	disable_mag = &disable_imu9;
	disable_gyr = &disable_imu9;
#endif
}

void init_imu_scaledStruct(imu_scaled_data* b1, enum ACCELEROMETER_SCALE a, enum MAGNETOMETER_SCALE m, enum GYRO_SCALE g, enum ACCEL_UNIT aU)
{
	make_disable_ptrs();
	b1->a = a;
	b1->g = g;
	b1->m = m;
	b1->u = aU;
}

void init_imu(imu_scaled_data* b1)
{
	make_disable_ptrs();
	delay_ms(2000);
#if defined(HAS_LSM303) && defined(HAS_LSM330)
	enum LSM303_ACCEL_SCALE scale;
	enum LSM330_ACCEL_SCALE scale1;

	switch(b1->a)
	{
		case SCALE_2G:
			scale = LSM303_SCALE_2_G;
			scale1 = LSM330_SCALE_2_G;
			break;
		case SCALE_4G:
			scale = LSM303_SCALE_4_G;
			scale1 = LSM330_SCALE_4_G;
			break;
		case SCALE_8G:
			scale = LSM303_SCALE_8_G;
			scale1 = LSM330_SCALE_8_G;
			break;
		default:
			return;
	}
#endif
#ifdef HAS_LSM9
	enum LSM9_ACCEL_SCALE scale;

	switch(b1->a)
	{
		case SCALE_2G:
			scale = LSM9_SCALE_2_G;
			break;
		case SCALE_4G:
			scale = LSM9_SCALE_4_G;
			break;
		case SCALE_8G:
			scale = LSM9_SCALE_8_G;
			break;
		default:
			return;
	}
#endif
	// delay_ms(200);
	// i2c_common_setup();
	delay_ms(1000);

	if (*disable_gyr == 0)
	{
		int a = gyro_setup();
		if (a < 0) *disable_gyr = 1;
	}
	if (*disable_gyr == 0)
	{
		int a = gyro_set_scale(b1->g);
		if (a < 0) *disable_gyr = 1;
	}
#if defined(HAS_LSM303) && defined(HAS_LSM330)
	if (disable_330 == 0)
	{
		int a = accel_lsm330_setup();
		if (a < 0) disable_330 = 1;
	}
	if (disable_330 == 0)
	{
		int a = accel_lsm330_set_scale(scale1);
		if (a < 0) disable_330 = 1;
	}
	if (disable_303 == 0)
	{
		delay_ms(200);
		int a = accel_lsm303_setup();
		if (a < 0) disable_303 = 1;
	}
	if (disable_303 == 0)
	{
		int a = accel_lsm303_set_scale(scale);
		if (a < 0) disable_303 = 1;
	}
#endif
#ifdef HAS_LSM9
	if (disable_imu9 == 0)
	{
		int a = accel_lsm9_setup();
		if (a < 0) disable_imu9 = 1;
	}
	if (disable_imu9 == 0)
	{
		int a = accel_lsm9_set_scale(scale);
		if (a < 0) disable_imu9 = 1;
	}
#endif
	if (*disable_mag == 0)
	{
		int a = mag_setup();
		if (a < 0) *disable_mag = 1;
	}
	if (*disable_mag == 0)
	{
		int a = mag_set_scale(b1->m);
		if (a < 0) *disable_mag = 1;
	}
}

int read_raw_imu_data(imu_raw_data* dataBuf)
{
	int a_data[3], r_data[3], m_data[3];
	uint8_t gyroData[6];
	uint8_t accelData[6];
	uint8_t magData[6];
#ifdef DUAL_ACCELEROMETER
	int a_330_data[3];
	uint8_t accel_330_data[6];
#endif

	if (*disable_gyr == 0)
	{
		gyro_read_all_data(gyroData);
	}
#if defined(HAS_LSM330)
	if (disable_330 == 0)
	{
		lsm330_accel_read_all_data(accel_330_data);
	}
#endif
#if defined(HAS_LSM303)
	if (disable_303 == 0)
	{
		lsm303_accel_read_all_data(accelData);
	}
#endif
#if defined(HAS_LSM9)
	if (disable_imu9 == 0)
	{
		lsm9_accel_read_all_data(accelData);
	}
#endif
	if (*disable_mag == 0)
	{
		mag_read_all_data(magData);
	}

	twosComp_sensorData_16b(gyroData, r_data);
#if defined(DUAL_ACCELEROMETER)
	twosComp_sensorData_16b(accel_330_data, a_330_data);
#endif
	twosComp_sensorData_16b(accelData, a_data);
	twosComp_sensorData_16b(magData, m_data);

	dataBuf->gyro_data[AXIS_X] = r_data[AXIS_X];
	dataBuf->gyro_data[AXIS_Y] = r_data[AXIS_Y];
	dataBuf->gyro_data[AXIS_Z] = r_data[AXIS_Z];

#if defined(DUAL_ACCELEROMETER)
	dataBuf->acc1_data[AXIS_X] = a_330_data[AXIS_X];
	dataBuf->acc1_data[AXIS_Y] = a_330_data[AXIS_Y];
	dataBuf->acc1_data[AXIS_Z] = a_330_data[AXIS_Z];
	dataBuf->acc2_data[AXIS_X] = a_data[AXIS_X];
	dataBuf->acc2_data[AXIS_Y] = a_data[AXIS_Y];
	dataBuf->acc2_data[AXIS_Z] = a_data[AXIS_Z];
#else
	dataBuf->acc1_data[AXIS_X] = a_data[AXIS_X];
	dataBuf->acc1_data[AXIS_Y] = a_data[AXIS_Y];
	dataBuf->acc1_data[AXIS_Z] = a_data[AXIS_Z];
#endif

	dataBuf->mag_data[AXIS_X] = m_data[AXIS_X];
	dataBuf->mag_data[AXIS_Y] = m_data[AXIS_Y];
	dataBuf->mag_data[AXIS_Z] = m_data[AXIS_Z];

	int a = 0;
#ifdef HAS_LSM303
	if (disable_303 > 0) a &= 0x01;
#endif
#ifdef HAS_LSM330
	if (disable_330 > 0) a &= 0x02;
#endif
#ifdef HAS_LSM9
	if (disable_imu9 > 0) a &= 0x04;
#endif
	if (a > 0) return -a; else return 1;
}

#ifdef HAS_LSM330
static void align_orientation_330(float *a)
{
	float a0 = a[0] , a1 = a[1] , a2 = a[2];
	//a[0] = -a1; a[1] = a0; a[2] = a2;
	a[0] = a0; a[1] = -a1; a[2] = -a2;
}
#endif

#ifdef HAS_LSM303
static void align_orientation_303(float *a)
{
	float a0 = a[0] , a1 = a[1] , a2 = a[2];
	//a[0] = a1; a[1] = -a0; a[2] = a2;
	a[0] = -a0; a[1] = a1; a[2] = -a2;
}
#endif

#ifdef HAS_LSM9
static void align_orientation_imu9(float *a)
{
	float a0 = a[0] , a1 = a[1] , a2 = a[2];
	a[0] = a0; a[1] = a1; a[2] = a2;
}
#endif

int init_imu_raw_data(imu_raw_data* r)
{
	int i;
	for (i=0; i<3; i++)
	{
		r->acc1_data[i] = 0;
#ifdef DUAL_ACCELEROMETER
		r->acc2_data[i] = 0;
#endif
		r->gyro_data[i] = 0;
		r->mag_data[i] = 0;
	}
}


int read_scaled_imu_data(imu_scaled_data* scaledBuf)
{
	float accel_multiplier = (scaledBuf->u == UNIT_G) ? (float)1 : (float)9.81;
	float gyro_scale_factor = 0.0f;
	float mag_scale_factor_xy = 0.0f;
	float mag_scale_factor_z = 0.0f;
	float accel_const = 0.0f;

#ifdef HAS_LSM330
	float accel_330_const = 0.0f;
#endif

#if defined(HAS_LSM303) && defined(HAS_LSM330)
	switch(scaledBuf->a)
	{
		case SCALE_2G:
			accel_const = accel_multiplier*LSM303_ACCEL_SCALE_FACTOR_2G;
			accel_330_const = accel_multiplier*LSM330_ACCEL_SCALE_FACTOR_2G;
			break;
		case SCALE_4G:
			accel_const = accel_multiplier*LSM303_ACCEL_SCALE_FACTOR_4G;
			accel_330_const = accel_multiplier*LSM330_ACCEL_SCALE_FACTOR_4G;
			break;
		case SCALE_8G:
			accel_const = accel_multiplier*LSM303_ACCEL_SCALE_FACTOR_8G;
			accel_330_const = accel_multiplier*LSM330_ACCEL_SCALE_FACTOR_8G;
			break;
		default:
			return;
	}
#endif
#if defined(HAS_LSM9)
	switch(scaledBuf->a)
	{
		case SCALE_2G:
			accel_const = accel_multiplier*LSM9_ACCEL_SCALE_FACTOR_2G;
			break;
		case SCALE_4G:
			accel_const = accel_multiplier*LSM9_ACCEL_SCALE_FACTOR_4G;
			break;
		case SCALE_8G:
			accel_const = accel_multiplier*LSM9_ACCEL_SCALE_FACTOR_8G;
			break;
		default:
			return -2;
	}
#endif

	switch(scaledBuf->g)
	{
		case SCALE_250_DPS:
			gyro_scale_factor = K_GYRO_250;
			break;
		case SCALE_500_DPS:
			gyro_scale_factor = K_GYRO_500;
			break;
		case SCALE_2000_DPS:
			gyro_scale_factor = K_GYRO_2000;
			break;
		default:
			return -2;
	}

#ifdef LSM303DLM
	switch(scaledBuf->m)
	{
	case SCALE_13GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_13_xy;
		mag_scale_factor_z = K_MAGNETOMETER_13_z;
		break;
	case SCALE_19GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_19_xy;
		mag_scale_factor_z = K_MAGNETOMETER_19_z;
		break;
	case SCALE_25GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_25_xy;
		mag_scale_factor_z = K_MAGNETOMETER_25_z;
		break;
	case SCALE_40GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_40_xy;
		mag_scale_factor_z = K_MAGNETOMETER_40_z;
		break;
	case SCALE_47GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_47_xy;
		mag_scale_factor_z = K_MAGNETOMETER_47_z;
		break;
	case SCALE_56GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_56_xy;
		mag_scale_factor_z = K_MAGNETOMETER_56_z;
		break;
	case SCALE_81GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_81_xy;
		mag_scale_factor_z = K_MAGNETOMETER_81_z;
		break;
	default:
		mag_scale_factor_xy = 0.0f;
		mag_scale_factor_z  = 0.0f;
		break;
	}
#endif
#if defined(LSM303D) || defined(LSM9DS0)
	switch(scaledBuf->m)
	{
	case SCALE_2GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_2_xy;
		mag_scale_factor_z = K_MAGNETOMETER_2_z;
		break;
	case SCALE_4GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_4_xy;
		mag_scale_factor_z = K_MAGNETOMETER_4_z;
		break;
	case SCALE_8GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_8_xy;
		mag_scale_factor_z = K_MAGNETOMETER_8_z;
		break;
	case SCALE_12GAUSS:
		mag_scale_factor_xy = K_MAGNETOMETER_12_xy;
		mag_scale_factor_z = K_MAGNETOMETER_12_z;
		break;
	default:
		mag_scale_factor_xy = 0.0f;
		mag_scale_factor_z  = 0.0f;
		break;
	}
#endif

	imu_raw_data rawBuffer;
	init_imu_raw_data(&rawBuffer);
	int a = read_raw_imu_data(&rawBuffer);
	if (a < 0) return a;

//	 static int n_send = 0;
//	 n_send ++;
//	if (n_send >= 1)
//	{
//	char s[64];
//	sprintf(s,"A1:%d,%d,%d\n",rawBuffer.acc1_data[AXIS_X],rawBuffer.acc1_data[AXIS_Y],rawBuffer.acc1_data[AXIS_Z]);
//	AP_send_msg(s);
//	sprintf(s,"G1:%d,%d,%d\n",rawBuffer.gyro_data[AXIS_X],rawBuffer.gyro_data[AXIS_Y],rawBuffer.gyro_data[AXIS_Z]);
//	AP_send_msg(s);
//	sprintf(s,"M1:%d,%d,%d\n",rawBuffer.mag_data[AXIS_X],rawBuffer.mag_data[AXIS_Y],rawBuffer.mag_data[AXIS_Z]);
//	AP_send_msg(s);
//	n_send = 0;
//	}


#if defined(DUAL_ACCELEROMETER)
	scaledBuf->acc1_data[AXIS_X] = accel_330_const * (float)rawBuffer.acc1_data[AXIS_X];
	scaledBuf->acc1_data[AXIS_Y] = accel_330_const * (float)rawBuffer.acc1_data[AXIS_Y];
	scaledBuf->acc1_data[AXIS_Z] = accel_330_const * (float)rawBuffer.acc1_data[AXIS_Z];
	align_orientation_330(scaledBuf->acc1_data);
	scaledBuf->acc2_data[AXIS_X] = accel_const * (float)rawBuffer.acc2_data[AXIS_X];
	scaledBuf->acc2_data[AXIS_Y] = accel_const * (float)rawBuffer.acc2_data[AXIS_Y];
	scaledBuf->acc2_data[AXIS_Z] = accel_const * (float)rawBuffer.acc2_data[AXIS_Z];
	align_orientation_303(scaledBuf->acc2_data);
#else
	scaledBuf->acc1_data[AXIS_X] = accel_const * (float)rawBuffer.acc1_data[AXIS_X];
	scaledBuf->acc1_data[AXIS_Y] = accel_const * (float)rawBuffer.acc1_data[AXIS_Y];
	scaledBuf->acc1_data[AXIS_Z] = accel_const * (float)rawBuffer.acc1_data[AXIS_Z];
	align_orientation_imu9(scaledBuf->acc1_data);
	scaledBuf->acc2_data[AXIS_X] = accel_const * (float)rawBuffer.acc1_data[AXIS_X];
	scaledBuf->acc2_data[AXIS_Y] = accel_const * (float)rawBuffer.acc1_data[AXIS_Y];
	scaledBuf->acc2_data[AXIS_Z] = accel_const * (float)rawBuffer.acc1_data[AXIS_Z];
	align_orientation_imu9(scaledBuf->acc2_data);
#endif

	scaledBuf->gyro_data[AXIS_X] = gyro_scale_factor * (float)rawBuffer.gyro_data[AXIS_X];
	scaledBuf->gyro_data[AXIS_Y] = gyro_scale_factor * (float)rawBuffer.gyro_data[AXIS_Y];
	scaledBuf->gyro_data[AXIS_Z] = gyro_scale_factor * (float)rawBuffer.gyro_data[AXIS_Z];
#if defined(HAS_LSM330)
	align_orientation_330(scaledBuf->gyro_data);
#elif defined(HAS_LSM9)
	align_orientation_imu9(scaledBuf->gyro_data);
#endif

	scaledBuf->mag_data[AXIS_X] = mag_scale_factor_xy * (float)rawBuffer.mag_data[AXIS_X];
	scaledBuf->mag_data[AXIS_Y] = mag_scale_factor_xy * (float)rawBuffer.mag_data[AXIS_Y];
	scaledBuf->mag_data[AXIS_Z] = mag_scale_factor_z  * (float)rawBuffer.mag_data[AXIS_Z];
#if defined(HAS_LSM303)
	align_orientation_303(scaledBuf->mag_data);
#elif defined(HAS_LSM9)
	align_orientation_imu9(scaledBuf->mag_data);
#endif
	//  static int n_send = 0;
	//  n_send ++;
	// if (n_send > 10)
	// {
	// char s[64];
	// sprintf(s,"A1:%.4f,%.4f,%.4f\n",scaledBuf->acc1_data[AXIS_X],scaledBuf->acc1_data[AXIS_Y],scaledBuf->acc1_data[AXIS_Z]);
	// AP_send_msg(s);
	// sprintf(s,"A2:%.4f,%.4f,%.4f\n",scaledBuf->acc2_data[AXIS_X],scaledBuf->acc2_data[AXIS_Y],scaledBuf->acc2_data[AXIS_Z]);
	// AP_send_msg(s);
	// sprintf(s,"G1:%.4f,%.4f,%.4f\n",scaledBuf->gyro_data[AXIS_X],scaledBuf->gyro_data[AXIS_Y],scaledBuf->gyro_data[AXIS_Z]);
	// AP_send_msg(s);
	// sprintf(s,"M1:%.4f,%.4f,%.4f\n",scaledBuf->mag_data[AXIS_X],scaledBuf->mag_data[AXIS_Y],scaledBuf->mag_data[AXIS_Z]);
	// AP_send_msg(s);
	// n_send = 0;
	// }
	return 1;
}

int to_signed_int_12(uint8_t low_byte, uint8_t high_byte) //@Warning: ASSUMES LEFT-ALIGNED DATA!!!
														  //@Verified
{
	uint32_t res = (high_byte<<4) | (low_byte>>4);
	return ((res & bit(11)) == bit(11)) ? (-1*(((~res) + 1) & 0xfff)) : res;
	/*
    union {
    	uint8_t data[2];
    	short int value;
    } output_val;
    output_val.data[0] = low_byte;
    output_val.data[1] = high_byte;
    return output_val.value;
    */
}

int to_signed_int_16(uint8_t low_byte, uint8_t high_byte) //@Warning: ASSUMES LEFT-ALIGNED DATA!!!
{
//   uint32_t res = (high_byte<<8) | low_byte;
//   return ((res & bit(15)) == bit(15)) ? (-1*(((~res) + 1) & 0x7fff)) : res;
    union {
    	uint8_t data[2];
    	short int value;
    } output_val;
    output_val.data[0] = low_byte;
    output_val.data[1] = high_byte;
    return output_val.value;
}

short int twosComp_16b(uint8_t low_byte, uint8_t high_byte)
{
	union {
		uint8_t data[2];
		short int value;
	} output_val;
	output_val.data[0] = low_byte;
	output_val.data[1] = high_byte;
	return output_val.value;
}

void twosComp_sensorData_16b(uint8_t data[6], int output[3])
{
	output[0] = (int)twosComp_16b(data[0], data[1]);
	output[1] = (int)twosComp_16b(data[2], data[3]);
	output[2] = (int)twosComp_16b(data[4], data[5]);
}

#ifdef HAS_LSM330
/*
 * Sets up LSM330DL accelerometer sensor. Turns on all three axes and sets them to sample at... ??????
 */

int accel_lsm330_setup(void)	//@Verified
{
    accel_lsm330_write_reg(0x20,0x77);
#ifdef LSM330DL
    accel_lsm330_write_reg(0x23,0x88);
#endif
#ifdef LSM330DLC
    accel_lsm330_write_reg(0x23,0x08);
#endif
	return 1;
}
int accel_lsm330_set_scale(enum LSM330_ACCEL_SCALE scale) //@Verified
{
    uint8_t original_byte = 0x88;
#ifdef LSM330DL
    original_byte = 0x88;
#endif
#ifdef LSM330DLC
    original_byte = 0x08;
#endif

    switch(scale)
    {
    case LSM330_SCALE_2_G:
    	original_byte |= 0;
    	break;
    case LSM330_SCALE_4_G:
    	original_byte |= bit(4);
    	break;
    case LSM330_SCALE_8_G:
    	original_byte |= bit(5);
    	break;
    case LSM330_SCALE_16_G:
    	original_byte |= bit(5) | bit(4);
    	break;
    default:
    	return -1;
    }

	int a = accel_lsm330_write_reg(0x23, original_byte);
	if (a < 0) return -1;
    return 1;
}
#endif

#ifdef HAS_LSM303
int accel_lsm303_setup(void) //@Verified
{
#ifdef LSM303DLM
    accel_lsm303_write_reg(0x20,0x3f);
#endif
#ifdef LSM303D
    accel_lsm303_write_reg(0x20,0x87);
#endif

#ifdef LSM303DLM
    accel_lsm303_write_reg(0x23,0x80);
#endif
  return 1;
}
int accel_lsm303_set_scale(enum LSM303_ACCEL_SCALE scale)
{

    uint8_t original_byte = 0x80;
#ifdef LSM303DLM
    original_byte = 0x80;
    switch(scale)
    {
    case LSM303_SCALE_2_G:
    	original_byte |= 0;
    	break;
    case LSM303_SCALE_4_G:
    	original_byte |= bit(4);
    	break;
    case LSM303_SCALE_8_G:
    	original_byte |= bit(5) | bit(4);
    	break;
    default:
    	return -1;
    }

	int a  = accel_lsm303_write_reg(0x23, original_byte);
	if (a < 0) return -1;
#endif
#ifdef LSM303D
    original_byte = 0x00;
    switch(scale)
    {
    case LSM303_SCALE_2_G:
    	original_byte |= 0;
    	break;
    case LSM303_SCALE_4_G:
    	original_byte |= bit(3);
    	break;
    case LSM303_SCALE_8_G:
    	original_byte |= bit(4) | bit(3);
    	break;
    default:
    	return -1;
    }
    original_byte |= bit(6); // set accelerometer anti-alias filter bandwidth to 194 Hz

	int a = accel_lsm303_write_reg(0x21, original_byte);
	if (a < 0) return -1;
#endif
    return 1;
}
#endif

#ifdef HAS_LSM9
int accel_lsm9_setup(void) //@Verified
{
    accel_lsm9_write_reg(0x20,0x8F); // 400 Hz update rate, output registers updated only after read
//    UARTprintf("Accel: 0x%02x\r\n", I2CReceive(ADDR_ACCEL_LSM9,0x20));
  return 1;
}
int accel_lsm9_set_scale(enum LSM9_ACCEL_SCALE scale)
{

    uint8_t original_byte = 0x00;
    switch(scale)
    {
    case LSM9_SCALE_2_G:
    	original_byte |= 0;
    	break;
    case LSM9_SCALE_4_G:
    	original_byte |= bit(3);
    	break;
    case LSM9_SCALE_8_G:
    	original_byte |= bit(4) | bit(3);
    	break;
    default:
    	return -1;
    }
    original_byte |= bit(6) | bit(7); // set accelerometer anti-alias filter bandwidth to 50 Hz

	int a = accel_lsm9_write_reg(0x21, original_byte);
	if (a < 0) return -1;
    return 1;
}
#endif

int mag_setup(void)	//@Verified
{
	int a;
#ifdef LSM303DLM
    a = mag_write_reg(0x01 , 0x20);
    if (a < 0) return -1;
    a = mag_write_reg(0x02 , 0x00);
    if (a < 0) return -1;
#endif
#if defined(LSM303D) || defined(LSM9DS0)
    a = mag_write_reg(0x26,0x00); // disable low power mode
    if (a < 0) return -1;
    a = mag_write_reg(0x24,0x70); // 50 Hz data rate, high resolution -- set highest bit 1 to enable temperature sensor
    if (a < 0) return -1;
#endif
    return 1;
}

int mag_set_scale(enum MAGNETOMETER_SCALE m)
{
	uint8_t scale_byte = 0x00;
#ifdef LSM303DLM
	switch(m)
	{
	case SCALE_13GAUSS:
		scale_byte = (1<<5);
		break;
	case SCALE_19GAUSS:
		scale_byte = (1<<6);
		break;
	case SCALE_25GAUSS:
		scale_byte = (1<<5 | 1<<6);
		break;
	case SCALE_40GAUSS:
		scale_byte = (1<<7);
		break;
	case SCALE_47GAUSS:
		scale_byte = (1<<7 | 1<<5);
		break;
	case SCALE_56GAUSS:
		scale_byte = (1<<7 | 1<<6);
		break;
	case SCALE_81GAUSS:
		scale_byte = (1<<7 | 1<<6 | 1<<5);
		break;
	default:
		return;
	}
	int a = mag_write_reg(0x01, scale_byte);
	if (a < 0) return -1;
#endif
#if defined(LSM303D) || defined(LSM9DS0)
	switch(m)
	{
	case SCALE_2GAUSS:
		scale_byte = 0;
		break;
	case SCALE_4GAUSS:
		scale_byte = (1<<5);
		break;
	case SCALE_8GAUSS:
		scale_byte = (1<<6);
		break;
	case SCALE_12GAUSS:
		scale_byte = ( (1<<5) | (1<<6) );
		break;
	default:
		return -2;
	}
	int a = mag_write_reg(0x25, scale_byte);
	if (a < 0) return -1;
#endif
	return 1;
}


int gyro_setup(void)
{
	int a = gyro_write_reg(0x20, 0xff);//(bit(2) | bit(1) | bit(0)));// Power on + enable all 3 sensing axes; 760 Hz output data rate and 100 Hz cut-off
	if (a < 0) return -1;
	#if defined(LSM330DLC) || defined(LSM9DS0)
	a = gyro_write_reg(0x21, 0x06); // high-pass filter; cut off frequency of 0.9 Hz (if ODR = 760 Hz as configured in write to 0x20)
	if (a < 0) return -1;
	//a = gyro_write_reg(0x24, 0x12); // enable high-pass filter
	a = gyro_write_reg(0x24, 0x00); // disable high-pass filter
	if (a < 0) return -1;
	#endif
	return 1;
}

int gyro_set_scale(enum GYRO_SCALE g)
{
	uint8_t scale_byte = 0;
#ifdef LSM330DL
        scale_byte = 1<<7;
#endif

	switch(g)
	{
	case SCALE_250_DPS:
		scale_byte |= 0x00;
		break;
	case SCALE_500_DPS:
		scale_byte |= (1<<4);
		break;
	case SCALE_2000_DPS:
		scale_byte |= (1<<5);
		break;
	default:
		return -2;
	}
#ifdef LSM9DS0
	scale_byte |= bit(7); // enable block data update -- output registers updated only after read
#endif
	int a = gyro_write_reg(0x23, scale_byte);
	if (a < 0) return -1;
	return 1;
}

#ifdef HAS_LSM303
int accel_lsm303_write_reg(uint8_t reg, uint8_t val)
{
    return i2c_write_reg_lsm(ADDR_ACCEL_LSM303,reg,val,&disable_303,f);
}
#endif

#ifdef HAS_LSM330
int accel_lsm330_write_reg(uint8_t reg, uint8_t val)
{
    return i2c_write_reg_lsm(ADDR_ACCEL_LSM330,reg,val,&disable_330,f);
}
#endif

#ifdef HAS_LSM9
int accel_lsm9_write_reg(uint8_t reg, uint8_t val)
{
    return i2c_write_reg_lsm(ADDR_ACCEL_LSM9,reg,val,f);
}
#endif

int mag_write_reg(uint8_t reg, uint8_t val)
{
#ifdef HAS_LSM303
    return i2c_write_reg_lsm(ADDR_MAG_LSM303,reg,val,&disable_303,f);
#elif defined(HAS_LSM9)
    return i2c_write_reg_lsm(ADDR_MAG_LSM9,reg,val,f);
#endif
}

int gyro_write_reg(uint8_t reg, uint8_t val) //@Verified
{
#ifdef HAS_LSM330
    return i2c_write_reg_lsm(ADDR_GYRO_LSM330,reg,val,f);
#elif defined(HAS_LSM9)
    return i2c_write_reg_lsm(ADDR_GYRO_LSM9,reg,val,f);
#endif
}

#ifdef HAS_LSM303
int lsm303_accel_read_all_data(uint8_t ret[6])
{
    return i2c_read_bytes(ADDR_ACCEL_LSM303,0xA8,6,ret,&disable_303);
}
#endif

#ifdef HAS_LSM330
int lsm330_accel_read_all_data(uint8_t ret[6])
{
    return i2c_read_bytes(ADDR_ACCEL_LSM330,0xA8,6,ret,&disable_330);
}
#endif

#ifdef HAS_LSM9
int lsm9_accel_read_all_data(uint8_t ret[6])
{
    //AP_send_msg("Reading acc");
    ret[0] = I2CReceive(ADDR_ACCEL_LSM9,0x28);
    ret[1] = I2CReceive(ADDR_ACCEL_LSM9,0x29);
    ret[2] = I2CReceive(ADDR_ACCEL_LSM9,0x2A);
	ret[3] = I2CReceive(ADDR_ACCEL_LSM9,0x2B);
	ret[4] = I2CReceive(ADDR_ACCEL_LSM9,0x2C);
	ret[5] = I2CReceive(ADDR_ACCEL_LSM9,0x2D);
    // return i2c_read_bytes(ADDR_ACCEL_LSM9,0x28,6,ret,&disable_imu9);
    return 1;
}
#endif

int mag_read_all_data(uint8_t ret[6])
{
#ifdef LSM303DLM
    uint8_t ret1[6];
    int a = i2c_read_bytes(ADDR_MAG_LSM303,0x83,6,ret1,&disable_303);
    ret[1] = ret1[0]; ret[0] = ret1[1];
    ret[5] = ret1[2]; ret[4] = ret1[3];
    ret[3] = ret1[4]; ret[2] = ret1[5];
#endif
#ifdef LSM303D
    int a = i2c_read_bytes(ADDR_MAG_LSM303,0x88,6,ret,&disable_303);
#endif
#ifdef LSM9DS0
    //AP_send_msg("Reading mag");
    ret[0] = I2CReceive(ADDR_MAG_LSM9,0x08);
    ret[1] = I2CReceive(ADDR_MAG_LSM9,0x09);
    ret[2] = I2CReceive(ADDR_MAG_LSM9,0x0A);
	ret[3] = I2CReceive(ADDR_MAG_LSM9,0x0B);
	ret[4] = I2CReceive(ADDR_MAG_LSM9,0x0C);
	ret[5] = I2CReceive(ADDR_MAG_LSM9,0x0D);
    // int a = i2c_read_bytes(ADDR_MAG_LSM9,0x08,6,ret,&disable_imu9);
#endif
	return 1;
}

int gyro_read_all_data(uint8_t ret[6])
{
#ifdef HAS_LSM330
    return i2c_read_bytes(ADDR_GYRO_LSM330,0xA8,6,ret,&disable_330);
#elif defined(HAS_LSM9)
    //AP_send_msg("Reading gyr");
    ret[0] = I2CReceive(ADDR_GYRO_LSM9,0x28);
    ret[1] = I2CReceive(ADDR_GYRO_LSM9,0x29);
    ret[2] = I2CReceive(ADDR_GYRO_LSM9,0x2A);
    ret[3] = I2CReceive(ADDR_GYRO_LSM9,0x2B);
    ret[4] = I2CReceive(ADDR_GYRO_LSM9,0x2C);
    ret[5] = I2CReceive(ADDR_GYRO_LSM9,0x2D);
    // return i2c_read_bytes(ADDR_GYRO_LSM9,0x28,6,ret,&disable_imu9);
#endif
    return 1;
}

