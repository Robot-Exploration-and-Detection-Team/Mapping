// Stub header for compilation without livox_ros_driver
// FAST_LIO uses this only for AVIA LiDAR type (lidar_type=1)
// For simulation we use MARSIM type (lidar_type=4) which doesn't need this
#pragma once

#include <stdint.h>
#include <vector>
#include <std_msgs/Header.h>
#include <boost/shared_ptr.hpp>

namespace livox_ros_driver {

struct CustomPoint {
    float x, y, z;
    float reflectivity;
    uint8_t tag;
    uint8_t line;
    uint32_t offset_time;
};

struct CustomMsg {
    std_msgs::Header header;
    uint64_t timebase;
    uint32_t point_num;
    uint8_t lidar_id;
    std::vector<CustomPoint> points;

    typedef boost::shared_ptr<CustomMsg> Ptr;
    typedef boost::shared_ptr<const CustomMsg> ConstPtr;
};

} // namespace livox_ros_driver