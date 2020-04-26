#include <iostream>
#include <caffe_deep_network.hpp>
#include <ees_engine.hpp>
#include <agent.hpp>
#include <csignal>

static bool stop_signal_called = false;
void sig_int_handler(int){stop_signal_called = true;}



void TestNetwork();
void LearnigCase1();
void Test2LevelLearning();
void TestWinAndPool();
void TestWideLearning();

int main()
{
    //LearnigCase1();
    Test2LevelLearning();
    //TestWideLearning();
   // TestWinAndPool();
}

void TestWinAndPool()
{
    CDNN::DataUnitSptr state = CDNN::DataUnitSptr(new CDNN::DataUnit<float> (1000, "cur_state"));

    float *sp=state->begin();
    for(auto i=0;i<10;i++)
    {
        for (auto j=0;j<100;j++)
        {
            sp[i*100+j]=j;
        }
    }

    state->Show(100);

    CDNN::DataUnitSptr state2=PoolState(state,10);

    state2->Show(10);

    CDNN::DataUnitSptr state3=WindowState(state,20,10,100);
    state3->Show(10);

}



void TestNetwork()
{


    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    CDNN::DQN dqn("dqn_solver.prototxt",50000,0.95);
    const int in_size=400;
    CDNN::Shape_t shape={32,1,20,20,2};
    dqn.Initialize(shape);
    std::cout<<"freq learning"<<std::endl;


    std::array<float, in_size> state1;
    std::array<float, in_size> state2;
    std::vector<std::array<float,in_size>> stateVec;

    for(auto i=0;i<in_size;i++)
    {
        if(i<in_size/2) {
            state1[i] = random()%20;
            state2[i] = random()%3;
        } else{
            state1[i]=random()%3;
            state2[i]=random()%20;
        }
    }

    stateVec.push_back(state1);
    stateVec.push_back(state2);

    int times=0;
    while(times<20000)
    {

        int state_index=times % 2;
        int next_state_index = (times+1) % 2;
        CDNN::DataUnitSptr cur_state = CDNN::DataUnitSptr(new CDNN::DataUnit<float> (in_size, "cur_state"));
        cur_state->Copy(&(stateVec[state_index][0]),0,in_size);
        float epl=1-times/15000;
        if(epl<0) epl=0;
        auto action=dqn.SelectAction(cur_state,epl);  //the object will pushed but will not be deleted as used by transition;






        float reward;
        if (state_index==action) reward=1.0; else reward=-1;
        std::cout<<"times="<<times<<" cur_state="<<state_index<<" action="<<action<<" reward="<<reward<<std::endl;

        CDNN::DataUnitSptr next_state = CDNN::DataUnitSptr(new CDNN::DataUnit<float> (in_size, "next_state"));
        next_state->Copy(&(stateVec[next_state_index][0]),0,in_size);
        const auto transition =
                CDNN::DQN::Transition(cur_state,action,reward,next_state);

        dqn.AddTransition(transition);

        if (dqn.memory_size()>=100)
            dqn.Update();

        times++;

    }
}


void LearnigCase1()
{
   //初始化操作
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    generateColorMap();

    //构建平台
    EES_Engine engine(200,20e6); //1000个频点，20MHz
    //构建感知设备
    Sensor sensor(200,&engine);



    //构建发射机

                       //start stop  step  band  alpha  power cycle     mode
    CommParam_t   cparam1{0e6, 1.8e6, 200e3, 1.8e6,   0.5,  80.0,   1, F_FIXED};
    CommParam_t   cparam2{5e6, 12e6,200e3, 7e6,   0.5,  80.0,   1,     F_FIXED};
    CommParam_t   cparam3{14e6,20e6,200e3, 6e6,   0.5,  80.0,   1,     F_FIXED};

    CommParam_t   cparam4{0e6, 20e6,  2e6, 2e6,   0.5,  20.0,  10,     F_SLAVE};

    Transmitter tx1(cparam1, &engine);
    Transmitter tx2(cparam2, &engine);
    Transmitter tx3(cparam3, &engine);
    Transmitter tx(cparam4, &engine);



    Receiver rx(&tx,&sensor,&engine,"fl_200x200_solver.prototxt");




    //注册设备
    engine.RegisterDevice((Device *)&tx1);
    engine.RegisterDevice((Device *)&tx2);
    engine.RegisterDevice((Device *)&tx3);
    engine.RegisterDevice((Device *)&tx);

    engine.RegisterDevice((Device *)&rx);

    engine.RegisterDevice((Device *)&sensor);


    std::signal(SIGINT, &sig_int_handler);
    //开始仿真
    while(not stop_signal_called)
    {
        engine.DriveSimlation();
    }

    std::cout<<" program stopted by ctrl c"<<std::endl;



}

std::ofstream outfile;
void Test2LevelLearning()
{
    outfile.open("record.txt");
    //初始化操作
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    generateColorMap();

    //构建平台
    EES_Engine engine(1000,20e6); //1000个频点，20MHz
    //构建感知设备
    Sensor sensor(200,&engine);



    //构建发射机

                       //start stop  step band  alpha power cycle mode
    CommParam_t   cparam1{0e6, 4e6,  4e6,  4e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam2{4e6, 8e6,  80e3, 0.5e6, 0.5,  80.0,   1,  F_SWEEP};
    CommParam_t   cparam3{8e6,12e6, 0.2e6, 0.2e6, 0.5,  80.0,  10,  F_RANDOM};
    CommParam_t   cparam4{12e6,16e6,0.4e6, 0.2e6, 0.5,  80.0, 100,  F_COM};
    CommParam_t   cparam5{16e6,20e6, 1e6,  1e6,   0.5,  80.0,  20,  F_RANDOM};
    CommParam_t   cparam {0e6,20e6, 0.2e6, 0.2e6, 0.5,  20.0,  10,  F_SLAVE};

/*
    CommParam_t   cparam1{0e6,5e6, 5e6,  5e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam2{5e6,10e6, 5e6,  5e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam3{10e6,15e6, 5e6,  5e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam4{15e6,20e6, 5e6,  5e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam5{0e6,20e6, 0.2e6, 0.2e6, 0.5,  20.0,  10,  F_SLAVE};*/


    Transmitter tx1(cparam1, &engine);
    Transmitter tx2(cparam2, &engine);
    Transmitter tx3(cparam3, &engine);
    ComJammer   tx4(cparam4, &engine);
    Transmitter tx5(cparam5, &engine);

    Transmitter tx(cparam, &engine);
    WBReceiver rx(&tx,&sensor,&engine,"band_solver.prototxt","channel_solver.prototxt");



    //注册设备
    engine.RegisterDevice((Device *)&tx1);
    engine.RegisterDevice((Device *)&tx2);
    engine.RegisterDevice((Device *)&tx3);
    engine.RegisterDevice((Device *)&tx4);
    engine.RegisterDevice((Device *)&tx5);
    engine.RegisterDevice((Device *)&tx);
    engine.RegisterDevice((Device *)&rx);
    engine.RegisterDevice((Device *)&sensor);




    std::signal(SIGINT, &sig_int_handler);
    //开始仿真
    while(not stop_signal_called)
    {
        engine.DriveSimlation();
    }

    std::cout<<" program stopted by ctrl c"<<std::endl;



}



void TestWideLearning()
{
    outfile.open("record.txt");
    //初始化操作
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    generateColorMap();

    //构建平台
    EES_Engine engine(1000,20e6); //1000个频点，20MHz
    //构建感知设备
    Sensor sensor(200,&engine);



    //构建发射机

    //start stop  step band  alpha power cycle mode
    CommParam_t   cparam1{0e6, 4e6,  4e6,  4e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam2{4e6, 8e6,  80e3, 0.5e6, 0.5,  80.0,   1,  F_SWEEP};
    CommParam_t   cparam3{8e6,12e6, 0.2e6, 0.2e6, 0.5,  80.0,  10,  F_RANDOM};
    CommParam_t   cparam4{12e6,16e6,0.4e6, 0.2e6, 0.5,  80.0, 100,  F_COM};
    CommParam_t   cparam5{16e6,20e6, 1e6,  1e6,   0.5,  80.0,  20,  F_RANDOM};
    CommParam_t   cparam {0e6,20e6, 0.2e6, 0.2e6, 0.5,  20.0,  10,  F_SLAVE};

/*
    CommParam_t   cparam1{0e6,5e6, 5e6,  5e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam2{5e6,10e6, 5e6,  5e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam3{10e6,15e6, 5e6,  5e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam4{15e6,20e6, 5e6,  5e6,   0.5,  80.0, 100,  F_SLOTED};
    CommParam_t   cparam5{0e6,20e6, 0.2e6, 0.2e6, 0.5,  20.0,  10,  F_SLAVE};*/


    Transmitter tx1(cparam1, &engine);
    Transmitter tx2(cparam2, &engine);
    Transmitter tx3(cparam3, &engine);
    ComJammer   tx4(cparam4, &engine);
    Transmitter tx5(cparam5, &engine);

    Transmitter tx(cparam, &engine);

    Receiver rx(&tx,&sensor,&engine,"fl_1000x200_solver.prototxt");


    //注册设备
    engine.RegisterDevice((Device *)&tx1);
    engine.RegisterDevice((Device *)&tx2);
    engine.RegisterDevice((Device *)&tx3);
    engine.RegisterDevice((Device *)&tx4);
    engine.RegisterDevice((Device *)&tx5);
    engine.RegisterDevice((Device *)&tx);
    engine.RegisterDevice((Device *)&rx);
    engine.RegisterDevice((Device *)&sensor);




    std::signal(SIGINT, &sig_int_handler);
    //开始仿真
    while(not stop_signal_called)
    {
        engine.DriveSimlation();
    }

    std::cout<<" program stopted by ctrl c"<<std::endl;



}