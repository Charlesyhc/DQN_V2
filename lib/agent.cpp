#include <agent.hpp>

#include <iostream>
#include <math.h>
#define pi 3.14159265

using namespace std;

static WFcolor ColorMap[256];

extern ofstream outfile;


Sensor::Sensor(uint32_t duration,EES_Engine *engine):Device(SENSOR,engine)
{
    _have_tx=false;
    _action_cycle=1;
    _duration=duration;

    _waterFallImage.create(_duration,_engine->SpectrumSize(),CV_8UC3);

    for(auto i=0;i<_duration;i++)
    {
        DoubleDataUnitSptr zero_spec=DoubleDataUnitSptr(new CDNN::DataUnit<double> (engine->SpectrumSize(), "zero vector"));
        CDNN::DataUnitSptr random_spec_db=CDNN::DataUnitSptr(new CDNN::DataUnit<float> (engine->SpectrumSize(), "zero vector"));
        zero_spec->FillZero();
        random_spec_db->FillRandom(20);
        _waterfall.push_back(zero_spec);
        _waterfall_db.push_back(random_spec_db);
    }

    _max_db=20;
    _min_db=-90;

    _scale=255.0/(_max_db-_min_db);
}

void Sensor::DoAction()
{

    DoubleDataUnitSptr cur_spectrum=_engine->Feedback(Gos(),1);

    CDNN::DataUnitSptr cur_spectrum_db=CDNN::DataUnitSptr(new CDNN::DataUnit<float> (_engine->SpectrumSize(), "cur spectrum db"));


    double *sp=cur_spectrum->begin();
    float *sp_db=cur_spectrum_db->begin();
    for(auto i=0;i<_engine->SpectrumSize();i++)
    {
        if(sp[i]>0) {
            float db = 10 * log10(sp[i]);

            if (db > _max_db) db = _max_db;
            if (db < _min_db) db = _min_db;
            sp_db[i] = (db - _min_db) * _scale - 127.0;
        } else sp_db[i]=_min_db;


    }





    _waterfall.push_back(cur_spectrum);
    _waterfall.pop_front();


    _waterfall_db.push_back(cur_spectrum_db);
    _waterfall_db.pop_front();




    Show();


}

void Sensor::Show()
{

    //瀑布图
    for(auto y=0;y<_duration;y++)
    {
        cv::Vec3b *raw=_waterFallImage.ptr<cv::Vec3b>(y);
        for(auto x=0;x<_waterFallImage.cols;x++)
        {
            float *line=_waterfall_db[_duration-y-1]->begin();
            WFcolor wf=bytesTovec3(floor(line[x]+127.0));
            raw[x]=cv::Vec3b(wf.b,wf.g,wf.r);
        }
    }


    cv::imshow("WaterFall",_waterFallImage);
    cv::waitKey(1);

}

CDNN::DataUnitSptr Sensor::GetState()
{
    uint32_t lines=_waterfall_db.size();
    uint32_t rows=_engine->SpectrumSize();


    CDNN::DataUnitSptr sensor_state = CDNN::DataUnitSptr(new CDNN::DataUnit<float> (lines*rows, "sensor_state"));

    uint32_t offset=0;
    for(auto i=0;i<lines;i++)
    {
        sensor_state->Copy((_waterfall_db[i])->begin(),offset,rows);
        offset=offset+rows;
    }

    //sensor_state->Show(200);
   // std::cout<<sensor_state->Sum()<<std::endl;
    //assert(sensor_state->Sum()>0.1);


    return sensor_state;


}





Receiver::Receiver(Transmitter *tx,Sensor *sensor,EES_Engine *engine,std::string protofile):Device(RECEIVER,engine)
{
    _tx=tx;
    _sensor=sensor;
    _have_tx=false;
    _action_cycle=_tx->CParam().action_cycle;
    _action_index=0;


    _dqn=CDNN::DQN::sptr(new CDNN::DQN(protofile,5000,0.95));

    CDNN::Shape_t shape={32,1,_engine->SpectrumSize(),_sensor->Duration(),_tx->LegalActions().size()};
    _dqn->Initialize(shape);

    _epl=1;

    DoAction();


}

void Receiver::DoAction()
{

    _next_state=_sensor->GetState();
    if(_cur_state.get()!=NULL)
    {
        const auto transition =
                CDNN::DQN::Transition(_cur_state,_action_index,_reward,_next_state);


        _dqn->AddTransition(transition);

        if (_dqn->memory_size()>=100) {
           float loss= _dqn->Update();
            std::cout<<"loss="<<loss<<" reward="<<_reward<<" itration="<<_dqn->Iteration()<<std::endl;
            outfile<<loss<<" "<<_reward<<std::endl;
        }
    }




    _cur_state=_sensor->GetState();

    _epl=_epl-0.0001;
    if(_epl<0) _epl=0;

    _last_action_index=_action_index;

    _action_index=_dqn->SelectAction(_cur_state,_epl);

    _reward=CaculateReward();
    std::cout<<"reward="<<_reward<<std::endl;
    _tx->DoAction(_action_index);


}

float Receiver::CaculateReward()
{


    double start_freq=_tx->LegalActions()[_action_index];
    double band=_tx->CParam().band;

    uint32_t start_index=start_freq/_engine->DeltaF();

    uint32_t num=band/_engine->DeltaF();
    assert((start_index+num)<=_engine->SpectrumSize());

    double whole_rx_energy=0;
    uint32_t size=_sensor->WaterFall().size();
    for (auto t=0;t<_action_cycle;t++) {
        double *sp=_sensor->WaterFall()[size-t-1]->begin();


        for (auto i = 0; i < num; i++) {
            whole_rx_energy=whole_rx_energy+sp[i+start_index];
        }
    }

    double sig_power_db=_tx->CParam().txPower;


    double sig_power=pow(10,sig_power_db/10);

    double rx_gain=1.0e-6;

    double sig_energy=sig_power*_action_cycle*rx_gain;


    double noise_energy=whole_rx_energy-sig_energy;

    std::cout<<sig_energy<<" "<<whole_rx_energy<<" "<<noise_energy<<" "<<_action_index<<std::endl;

    bool jammed=false;

    if(noise_energy>0) {

        double snr = sig_energy / noise_energy;

        if (snr < 10) jammed = true;
    }


    if(jammed)
    {
        if(_action_index==_last_action_index) return -0.5;
        else return -1.5;

    } else
    {
        if(_action_index==_last_action_index) return 1.5;
        else return 0.5;
    }




}

WBReceiver::WBReceiver(Transmitter *tx, Sensor *sensor, EES_Engine *engine,std::string net1,std::string net2):Device(WB_RECEIVER,engine)
{
    _tx=tx;
    _sensor=sensor;
    _have_tx=false;
    _action_cycle=_tx->CParam().action_cycle;
    _action_index=0;


    _sub_band_num=10;

    _compress_rate=10; //10倍的频谱压缩

    _sub_band_size=_engine->SpectrumSize()/_sub_band_num;
    _channel_num_of_subband=tx->LegalActions().size()/_sub_band_num;


    _dqn_band=CDNN::DQN::sptr(new CDNN::DQN(net1,5000,0.95));

    CDNN::Shape_t shape1={32,1,_sensor->Duration(),_engine->SpectrumSize()/_compress_rate,_sub_band_num};

    _dqn_band->Initialize(shape1);





    _dqn_channel=CDNN::DQN::sptr(new CDNN::DQN(net2,5000,0.95));
    CDNN::Shape_t shape2={32,1,_sensor->Duration(),_sub_band_size,_channel_num_of_subband};
    _dqn_channel->Initialize(shape2);




    _epl=1;
    _last_action_index=0;
    _action_index=0;

    DoAction();



}




void WBReceiver::DoAction()
{

    _next_state=_sensor->GetState();



    if(_cur_state.get()!=NULL)
    {


        CDNN::DataUnitSptr cur_pool_state=PoolState(_cur_state,_compress_rate);
        CDNN::DataUnitSptr cur_win_state=WindowState(_cur_state,_action_band_index*_sub_band_size,_sub_band_size,_engine->SpectrumSize());


        CDNN::DataUnitSptr next_pool_state=PoolState(_next_state,_compress_rate);
        CDNN::DataUnitSptr next_win_state=WindowState(_next_state,_action_band_index*_sub_band_size,_sub_band_size,_engine->SpectrumSize());




        const auto transition1 =
                CDNN::DQN::Transition(cur_pool_state,_action_band_index,_reward,next_pool_state);

        const auto transition2 =
                CDNN::DQN::Transition(cur_win_state,_action_channel_index,_reward,next_win_state);


        _dqn_band->AddTransition(transition1);
        _dqn_channel->AddTransition(transition2);

        if (_dqn_band->memory_size()>=100)
        {
           float loss_band= _dqn_band->Update();
           float loss_channel= _dqn_channel->Update();

           std::cout<<"loss band="<<loss_band<<" loss_channle="<<loss_channel<<" reward="<<_reward<<" "<<_dqn_band->Iteration()<<std::endl;
           outfile<<loss_band<<" "<<loss_channel<<" "<<_reward<<std::endl;

        }
    }




    _cur_state=_sensor->GetState();
    CDNN::DataUnitSptr  cur_pool_state=PoolState(_cur_state,_compress_rate);





    _epl=_epl-0.0001;
    if(_epl<0) _epl=0;

    _action_band_index=_dqn_band->SelectAction(cur_pool_state,_epl);

    CDNN::DataUnitSptr  cur_win_state=WindowState(_cur_state,_action_band_index*_sub_band_size,_sub_band_size,_engine->SpectrumSize());

    _action_channel_index=_dqn_channel->SelectAction(cur_win_state,_epl);



    _last_action_index=_action_index;
    _action_index=_action_band_index*_channel_num_of_subband+_action_channel_index;


    _reward=CaculateReward();


    _tx->DoAction(_action_index);


}

float WBReceiver::CaculateReward()
{


    double start_freq=_tx->LegalActions()[_action_index];
    double band=_tx->CParam().band;

    uint32_t start_index=start_freq/_engine->DeltaF();

    uint32_t num=band/_engine->DeltaF();
    assert((start_index+num)<=_engine->SpectrumSize());

    double whole_rx_energy=0;
    uint32_t size=_sensor->WaterFall().size();
    for (auto t=0;t<_action_cycle;t++) {
        double *sp=_sensor->WaterFall()[size-t-1]->begin();


        for (auto i = 0; i < num; i++) {
            whole_rx_energy=whole_rx_energy+sp[i+start_index];
        }
    }

    double sig_power_db=_tx->CParam().txPower;


    double sig_power=pow(10,sig_power_db/10);

    double rx_gain=1.0e-6;

    double sig_energy=sig_power*_action_cycle*rx_gain;


    double noise_energy=whole_rx_energy-sig_energy;

    //std::cout<<sig_energy<<" "<<whole_rx_energy<<" "<<noise_energy<<" "<<_action_index<<std::endl;

    bool jammed=false;

    if(noise_energy>0) {

        double snr = sig_energy / noise_energy;

        if (snr < 10) jammed = true;
    }


    if(jammed)
    {
        if(_action_index==_last_action_index) return -0.5;
        else return -1.5;

    } else
    {
        if(_action_index==_last_action_index) return 1.5;
        else return 0.5;
    }





}



Transmitter::Transmitter(CommParam_t cparam, EES_Engine *engine):Device(TRANSMITTER,engine)
{
    _have_tx=true;
    _commu_param=cparam;

    _action_cycle=_commu_param.action_cycle;




    assert(_commu_param.step>0);
    uint32_t  action_num =(_commu_param.stop-_commu_param.start-_commu_param.band)/_commu_param.step +1;
    for(auto i=0;i<action_num;i++)
    {
        this->_legal_actions.push_back(i*_commu_param.step+_commu_param.start);
    }

    _action_index=0;


    this->GenerateBaseSig();
    DoAction();

}





void Transmitter::GenerateBaseSig()
{

    std::vector<double> Rcos_base=Rcosine(_commu_param.band, _engine->DeltaF(), _commu_param.alpha, _commu_param.txPower);
    _baseSig=DoubleDataUnitSptr(new CDNN::DataUnit<double> (Rcos_base.size(),"base sig"));
    _baseSig->Copy(&Rcos_base[0],0,Rcos_base.size());
   // _baseSig->Show(100);


}
ComJammer::ComJammer(CommParam_t cparam, EES_Engine *engine):Transmitter(cparam,engine)
{
    this->GenerateBaseSig();

}
void ComJammer::GenerateBaseSig()
{
    std::vector<double> Rcos_base=Rcosine(_commu_param.band, _engine->DeltaF(), _commu_param.alpha, _commu_param.txPower);
    uint32_t num=(_commu_param.stop-_commu_param.start)/_engine->DeltaF();

    _sig1=DoubleDataUnitSptr(new CDNN::DataUnit<double> (num, "sig1"));

    _sig2=DoubleDataUnitSptr(new CDNN::DataUnit<double> (num, "sig2"));



    double start_freq=0;
    while(true)
    {
        uint32_t start_pos=start_freq/_engine->DeltaF();

        int tail=num-start_pos;
        if(tail<=0) break;


        uint32_t len=min((uint32_t)Rcos_base.size(),num-start_pos);

        _sig1->Copy(&Rcos_base[0],start_pos,len);

        start_freq=start_freq+_commu_param.step;

    }



    start_freq=_commu_param.band;
    while(true)
    {
        uint32_t start_pos=start_freq/_engine->DeltaF();

        int tail=num-start_pos;
        if(tail<=0) break;


        uint32_t len=min((uint32_t)Rcos_base.size(),num-start_pos);

        _sig2->Copy(&Rcos_base[0],start_pos,len);

        start_freq=start_freq+_commu_param.step;

    }


    _action_index=0;


}

void ComJammer::DoAction()
{


    if(_action_index==0)
    {
        _baseSig=_sig1;
        _action_index=1;

    } else{

        _baseSig=_sig2;

        _action_index=0;

    }


}

void Transmitter::DoAction()
{



    uint32_t action_index=random() % _legal_actions.size();

    switch(_commu_param.fmode){
        case F_FIXED: break;
        case F_RANDOM:
        {
            _action_index=random() % _legal_actions.size();

        }break;
        case F_SWEEP:
        {
            _action_index=_action_index+1;
            if(_action_index>=_legal_actions.size())
            {
                _action_index=0;
            }
        }break;
        case F_LEARNING:break;

        case F_SLOTED:
        {
            _action_index=0;
            int seed=random() %100;
            if(seed>50) _have_tx=not _have_tx; //模拟时系通信过程
        }
        break;
        case F_SLAVE:
        {
            return; //接收端已经完成了

        }
    }
    if(_action_index<_legal_actions.size())
        _cur_freq=_legal_actions[_action_index];
   // std::cout<<_cur_freq<<std::endl;



}


//SLAVE 模式
void Transmitter::DoAction(uint32_t action_index)
{
    if(action_index<_legal_actions.size())
        _cur_freq=_legal_actions[action_index];
}




std::vector<double> Rcosine(double band, double delta_f, float alpha, float power)
{

    double half_band=band/2;
    uint32_t N=round(half_band/delta_f);

    double FN=half_band/(1+alpha);

    double F_Delta=half_band-FN;

    double F1=FN-F_Delta;

    double F2=FN+F_Delta;

    double Ts=1/(2*FN); //符号速率

    double HF[N];
    double _acc_HF=0;
    for (auto i=0;i<N;i++)
    {
        double fi=i*delta_f;
        if(fi<F1)
        {
            HF[i]=1.0;
        } else if(fi<F2)
        {
            double temp=0.5*(1+sin( Ts/(2*alpha)*(pi/Ts-2*pi*fi)));
            HF[i]=temp*temp;
        } else HF[i]=0;

        _acc_HF=_acc_HF+HF[i];

    }
    _acc_HF=_acc_HF*2;

    double power_dec=pow(10,(power/10));

    double scale=power_dec/_acc_HF;

    std::vector<double> spectrum;

    for (auto i=0;i<N;i++)
    {
        spectrum.push_back(HF[N-i-1]*scale);
    }

   // spectrum.push_back(scale);

    for (auto i=0;i<N;i++)
    {
        spectrum.push_back(HF[i]*scale);
    }


    return spectrum;


}


void generateColorMap()
{
    const WFcolor red={255,0,0};
    const WFcolor yellow={255,255,0};
    const WFcolor green={0,255,0};
    const WFcolor cyan={0,255,255};
    const WFcolor blue={0,0,255};
    const WFcolor cp[5]={blue,cyan,green,yellow,red};

    uint8_t r,g,b;
    WFcolor start,stop;
    int steps;
    int range=64;
    for(auto byte=0;byte<256;byte++)
    {
        int pos=byte/range+1;

        start=cp[pos-1];
        stop=cp[pos];
        steps=byte-(pos-1)*range;
        ColorMap[byte].r=start.r+(stop.r-start.r)*steps/range;
        ColorMap[byte].g=start.g+(stop.g-start.g)*steps/range;
        ColorMap[byte].b=start.b+(stop.b-start.b)*steps/range;
    }


}

WFcolor bytesTovec3(uint8_t byte)
{
    return ColorMap[byte];
}

CDNN::DataUnitSptr PoolState(CDNN::DataUnitSptr &state,uint32_t rate)
{

    CDNN::DataUnitSptr poolstate=CDNN::DataUnitSptr(new CDNN::DataUnit<float>(state->size()/rate,"pool state"));


    float *sp1=state->begin();
    float *sp2=poolstate->begin();
    uint32_t k=0;
    for (auto i=0;i<poolstate->size();i++)
    {
        float temp=0.0;

        for(auto j=0;j<rate;j++)
        {
            temp=temp+sp1[k];
            k=k+1;
        }

        sp2[i]=temp/rate;
    }


    return poolstate;

}


CDNN::DataUnitSptr WindowState(CDNN::DataUnitSptr &state,uint32_t start, uint32_t width, uint32_t owidth)
{

    uint32_t lines=state->size()/owidth;

    CDNN::DataUnitSptr winstate=CDNN::DataUnitSptr(new CDNN::DataUnit<float>(width*lines,"window state"));



    float *sp1=state->begin();
    float *sp2=winstate->begin();

    for (auto i=0;i<lines;i++)
    {
        for(auto j=0;j<width;j++)
        {
            sp2[i*width+j]=sp1[i*owidth+j+start];


        }


    }


    return winstate;

}