#include <ees_engine.hpp>
#include <math.h>
#include <algorithm>



Device::Device(DeviceType d_type, EES_Engine *engine)
{
    _engine=engine;
    _action_timer=0;


}

Device::~Device()
{

}

void Device::Work()
{
    _action_timer++;
    if(_action_timer==_action_cycle)
    {
        DoAction();
        _action_timer=0;
    }

}







EES_Engine::EES_Engine(uint32_t spectrum_size,float whole_band):_spectrum_size(spectrum_size),_whole_band(whole_band)
{
    _delta_f=_whole_band/_spectrum_size;
    GenerateNoise();
}

EES_Engine::~EES_Engine()
{

}

double free_trans_gain(Gos_t rx_gos, Gos_t tx_gos, float rx_gain)
{
    //待完善
    return  1.0e-6;
}




//从某位置感知频谱状态
DoubleDataUnitSptr EES_Engine::Feedback(Gos_t gos,float rx_gain)
{
    //要有多线程保护
    DoubleDataUnitSptr spectrum= DoubleDataUnitSptr(new CDNN::DataUnit<double> (_spectrum_size, "feedback spectrum"));
    spectrum->FillZero();
    double *rsp=spectrum->begin();
    for(auto i=0;i<_dev_list.size();i++)
    {
        Device* dev= _dev_list[i];

        if (dev->HaveTx()==false) continue;  //没有信号发送的设备跳过

        double gain=free_trans_gain(gos,dev->Gos(),rx_gain);



        int offset= dev->Freq() /_delta_f;

        //offset=offset+random() % 2;
        if(offset>=(int)_spectrum_size) continue;

        if(offset>=0) {
            uint32_t len = min(_spectrum_size - offset, dev->BaseSig()->size());
            //_sig_spectrum->Copy(_baseSig->begin(),offset,len);

            double *sp = dev->BaseSig()->begin();
            for (auto j = 0; j < len; j++) {
                rsp[j + offset] = rsp[j + offset] + sp[j] * gain;
            }
        } else{

            int len=dev->BaseSig()->size()+offset;
            if(len>0)
            {
                double *sp = dev->BaseSig()->begin();
                for (auto j = 0; j < len; j++) {
                    rsp[j ] = rsp[j] + sp[j-offset] * gain;
                }

            }

        }
    }

    uint32_t noise_index=random() &(NPS-1);
    auto noise=_noise_pool[noise_index];
    double *np=noise->begin();
    for(auto j=0;j<_spectrum_size;j++)
    {
        rsp[j]=rsp[j]+np[j]; //添加环境噪声
    }
    return spectrum;

}

void EES_Engine::GenerateNoise()
{

    double noise_level_db=-80;

    double noise_level=pow(10,noise_level_db/10);
    noise_level=0;

    for(auto k=0;k<this->NPS;k++) {
        DoubleDataUnitSptr noise = DoubleDataUnitSptr(new CDNN::DataUnit<double>(_spectrum_size, "noise"));
        double *np = noise->begin();
        for (auto i = 0; i < _spectrum_size; i++) {
            double x = (rand() & 0xffff)/65536.0;
            np[i] =sqrt(-2 * log(x)) * noise_level;
        }
        //noise->Show(100);
        _noise_pool.push_back(noise);
    }








}


void EES_Engine::RegisterDevice(Device *dev)
{
    this->_dev_list.push_back(dev);
    dev->SetID(_dev_list.size());

}


void EES_Engine::DriveSimlation()
{

    for(auto iter=_dev_list.begin();iter!=_dev_list.end();iter++)
    {
        (*iter)->Work();
    }

}



