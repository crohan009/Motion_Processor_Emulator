//#ifndef _CLOSEDCUBE_LPS25HB_h
//
//#define _CLOSEDCUBE_LPS25HB_h

void begin(uint8_t address);

void whoAmI();
float readTemperature();
double readT(); // short-cut for readTemperature

float readPressure();
double readP(); // short-cut for readPressure

float altitude(float pressure);

float cal_LPS25();
float avg();
