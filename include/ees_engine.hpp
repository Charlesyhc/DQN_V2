/*
 * 电磁环境模拟程序，从频域感知的角度模拟电磁环境
 * 基本框架是信号源序列 electromagnetic environment simulation
 */


#ifndef EES_ENGINE_HPP
#define EES_ENGINE_HPP


#include <caffe_deep_network.hpp>  //主要是为使用caffe 头文件中定义的数据结构


#define min(a,b)  (((a) < (b)) ? (a) : (b))


using DoubleDataUnitSptr= CDNN::DataUnit<double>::sptr;
//信号源位置
typedef struct {
    float x;
    float y;
    float h;

}Gos_t;

//信号源描述信息
typedef struct{
    uint32_t sID;//信号源身份信息
    Gos_t gos; //信号源位置
    DoubleDataUnitSptr spectrum; //信号源频谱

}sig_source_t;


enum DeviceType
{
    SENSOR=0x01,//感知设备
    TRANSMITTER=0x02, //发射机,也可以是干扰设备
    RECEIVER=0x03, //接收机
    WB_RECEIVER=0x04  //
};

class EES_Engine;

//模拟设备
class Device{
public:
    typedef boost::shared_ptr<Device> sptr;
    typedef std::vector<double> ActionSet;
protected:
    uint32_t _id; //设备身份编码
    Gos_t _gos; //设备所处的位置
    DeviceType _device_type; //设备类型
    bool _have_tx; //是否有发射设
    ActionSet _legal_actions; //设备的可选决策集合

    uint32_t _action_timer; //计时器
    uint32_t _action_cycle;//决策的周期
    EES_Engine *_engine;  //环境

    DoubleDataUnitSptr _baseSig; //基带信号，只有发送设备才有
    float _cur_freq; //中心频率

public:
    Device(DeviceType d_type, EES_Engine *engine);
    ~Device();

    bool HaveTx(){return _have_tx;}

    void SetID(uint32_t id){_id=id;};


    float Freq(){return _cur_freq;}


    void Work();


    DoubleDataUnitSptr BaseSig(){return _baseSig;}

    Gos_t Gos(){return _gos;}

    void AddActions(float action)
    {
        _legal_actions.push_back(action);
    };

    virtual void DoAction()=0;





};


//电磁环境模拟类
class EES_Engine
{
protected:
    std::vector<Device*> _dev_list;

    uint32_t _timmer;

    float _whole_band;// 整个带宽
    uint32_t _spectrum_size; //整个带宽的频点数
    float _delta_f;// 频率分辨率


    static const uint32_t NPS=256;
    std::vector<DoubleDataUnitSptr> _noise_pool; //噪声池



public:
    EES_Engine(uint32_t spectrum_size,float whole_band);
    ~EES_Engine();

    float WholeBand(){return _whole_band;}
    float DeltaF(){return _delta_f;}

    //增加设备
    void RegisterDevice(Device *dev);
    void DriveSimlation();
    void GenerateNoise();
    uint32_t SpectrumSize(){return _spectrum_size;}
    DoubleDataUnitSptr Feedback(Gos_t gos,float rx_gain);




};












#endif