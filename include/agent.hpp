#ifndef AGENT_HPP
#define AGENT_HPP


#include <caffe_deep_network.hpp>
#include <ees_engine.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>
#include <iostream>

struct WFcolor
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

void generateColorMap();
WFcolor bytesTovec3(uint8_t byte);
std::vector<double> Rcosine(double band, double delta_f, float alpha, float power);


CDNN::DataUnitSptr WindowState(CDNN::DataUnitSptr &state,uint32_t start, uint32_t width, uint32_t owidth);
CDNN::DataUnitSptr PoolState(CDNN::DataUnitSptr &state,uint32_t rate);



enum FreqMode_t
{
    F_RANDOM=0x01,  //随机
    F_FIXED=0x02,   //固定
    F_LEARNING=0x03,//基于学习
    F_SWEEP=0x04,   //扫频
    F_SLOTED=0x05,  //时系
    F_SLAVE=0x06,  //由发送端决定，此时需要和发送端配置相同参数
    F_COM=0x07

};


struct CommParam_t
{
    double start; //初始频率
    double stop; //结束频率
    double step; //频率步进
    double band;  //信号带宽

    float alpha; //滚降系数
    float txPower; //发送功率
    uint32_t action_cycle;
    FreqMode_t fmode;
};





//感知设备
class Sensor:protected Device
{
protected:
    uint32_t _duration;
    std::deque<DoubleDataUnitSptr> _waterfall;  //用于计算
    std::deque<CDNN::DataUnitSptr> _waterfall_db; //用于显示和学习
    cv::Mat _waterFallImage;

    float _max_db; //显示的最大dBm值
    float _min_db; //显示的最小dBm值

    float _scale;
public:
    Sensor(uint32_t duration, EES_Engine *engine);
    virtual void DoAction();
    const std::deque<DoubleDataUnitSptr> &WaterFall(){ return _waterfall;}
    void Show();
    uint32_t Duration(){ return _duration;}
    CDNN::DataUnitSptr  GetState();

};




//发送设备
class Transmitter:protected Device {
protected:
    CommParam_t _commu_param;
    uint32_t _action_index;
public:
    Transmitter(CommParam_t cparam, EES_Engine *engine);

    virtual void GenerateBaseSig();
    virtual void DoAction();
    void DoAction(uint32_t action_index);

    const ActionSet& LegalActions(){return _legal_actions;}
    const CommParam_t& CParam(){return _commu_param;}
    DoubleDataUnitSptr BaseSig(){return _baseSig;}

};


class ComJammer:protected Transmitter{
protected:
    DoubleDataUnitSptr _sig1;
    DoubleDataUnitSptr _sig2;

public:
    ComJammer(CommParam_t cparam, EES_Engine *engine);
   virtual void GenerateBaseSig();

   virtual void DoAction();



};





//接收设备，通信频率由接收设备来确定
class Receiver:protected Device {
protected:
    Transmitter *_tx; //与其配对的发送设备
    Sensor *_sensor;

    uint32_t _action_index;
    uint32_t _last_action_index;
    CDNN::DataUnitSptr _cur_state;
    CDNN::DataUnitSptr _next_state;
    float _reward;




    CDNN::DQN::sptr _dqn;

    float _epl;

public:
    Receiver(Transmitter *tx, Sensor *sensor, EES_Engine *engine,std::string protofile);

    virtual void DoAction();
    const uint32_t ActionIndex(){return _action_index;}

    float CaculateReward();



};




//宽带接收机
class WBReceiver:protected Device {
protected:
    uint32_t _sub_band_num; //划分的子带个数
    uint32_t _sub_band_size; //子带的大小


    uint32_t _action_band_index; //频带决策
    uint32_t _channel_num_of_subband;  //子带信道数量
    uint32_t _action_channel_index; //信道决策

    CDNN::DQN::sptr _dqn_band;
    CDNN::DQN::sptr _dqn_channel;


    Transmitter *_tx; //与其配对的发送设备
    Sensor *_sensor;

    uint32_t _last_action_index;
    uint32_t _action_index;
    CDNN::DataUnitSptr _cur_state;
    CDNN::DataUnitSptr _next_state;
    float _reward;
    float _epl;
    uint32_t _compress_rate; //频谱的压缩比例

public:
    WBReceiver(Transmitter *tx, Sensor *sensor, EES_Engine *engine,std::string net1,std::string net2);

    virtual void DoAction();
    const uint32_t ActionIndex(){return _action_index;}

    float CaculateReward();


};





















#endif