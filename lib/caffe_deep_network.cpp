
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>
#include <caffe_deep_network.hpp>


using namespace CDNN;


template <typename Dtype>
bool HasBlobSize(
        const caffe::Blob<Dtype>& blob,
        const int num,
        const int channels,
        const int height,
        const int width)
{
    std::cout<<blob.num()<<" "<< blob.channels()<<" "<<blob.height()<<" "<<blob.width()<<std::endl;
    return blob.num() == num &&
           blob.channels() == channels &&
           blob.height() == height &&
           blob.width() == width;
}

void DQN::Initialize(const Shape_t &shape)
{
    caffe::SolverParameter solver_param;

    caffe::ReadSolverParamsFromTextFileOrDie(_solver_param, &solver_param);

    _solver.reset(::caffe::SolverRegistry<float>::CreateSolver(solver_param)); //constructor

    if (solver_param.solver_mode()==0)
        std::cout<<"CPU MODE"<<std::endl;
    else
        std::cout<<"GPU MODE"<<std::endl;




    _net = _solver->net();

    // 或者Q值表
    _q_values_blob = _net->blob_by_name("q_values");
    _loss=_net->blob_by_name("loss");

    // dummy 数据初始化
    _dummy_input_data=DataUnitSptr(new DataUnit<float> (shape.batch*shape.output,"_dummy_input_data"));
    _dummy_input_data->FillZero();

    // 获得输入口
    _state_input_layer =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("state_input_layer"));
    assert(_state_input_layer);

    //格式确认
    assert(HasBlobSize(*_net->blob_by_name("state"),
                       shape.batch,
                       shape.channels,
                       shape.height,
                       shape.width));



    _target_input_layer =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("target_input_layer"));
    assert(_target_input_layer);

    //格式确认
    assert(HasBlobSize(*_net->blob_by_name("target"), shape.batch, shape.output, 1, 1));

    _filter_input_layer =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("filter_input_layer"));
    assert(_filter_input_layer);

    //格式确认
    assert(HasBlobSize(
            *_net->blob_by_name("filter"), shape.batch, shape.output, 1, 1));

    _shape=shape;
    _in_size=_shape.channels*_shape.width*_shape.height;
    _in_size_batch=_in_size*_shape.batch;

}


//epsilon 随机选择策略还是贪婪选择策略的概率
DQN::Action DQN::SelectAction(DataUnitSptr& state, double epsilon){

    assert(epsilon >= 0.0 && epsilon <= 1.0);
    Action action;
    if (std::uniform_real_distribution<>(0.0, 1.0)(_random_engine) < epsilon) {
        //按照均匀分布随机选择策略
        const auto random_idx =std::uniform_int_distribution<int>(0, _shape.output - 1)(_random_engine);
        action = random_idx;
        //std::cout<<"random action"<<action<<std::endl;

    } else {
        action = SelectActionGreedily(state).first; //which return is <action reward>, first get action
       // std::cout<<"greedy action"<<action<<std::endl;

    }

    //std::cout << " epsilon:" << epsilon << std::endl;
    return action;
}



DQN::AQPair DQN::SelectActionGreedily(DataUnitSptr& state)
{

    assert(state->size()==_in_size);

    StateBatch stateBatch;

    stateBatch.push_back(state);


    return SelectActionGreedilyBatch(stateBatch).front();



}

template <typename dType>
std::string VectorToStr(const std::vector<dType>& vec)
{
    assert(!vec.empty());

    std::ostringstream q_values_buf;

    for (auto i = 0; i < vec.size(); ++i) {
        const auto q_str = std::to_string(vec[i]);
        q_values_buf << q_str<<" ";
    }
    q_values_buf << std::endl;
    return q_values_buf.str();
}


std::vector<DQN::AQPair> DQN::SelectActionGreedilyBatch(StateBatch& stateBatch)
{
    DataUnitSptr stateBatchInput = DataUnitSptr(new DataUnit<float> (_in_size_batch, "stateBatchInput"));

    if(stateBatch.size()==1)
    {
        stateBatchInput->FillZero();
        stateBatchInput->Copy(stateBatch[0]->begin(),0,_in_size);

    }
    else
    {
        assert(stateBatch.size()==_shape.batch);

        for(auto i=0;i<_shape.batch;i++)
        {
            stateBatchInput->Copy(stateBatch[i]->begin(),_in_size*i,_in_size);
           // stateBatch[i]->Show(100);
        }
        //stateBatchInput->Show(100);

    }



    InputDataIntoLayers(stateBatchInput,_dummy_input_data,_dummy_input_data);
    float loss;
    _net->Forward(&loss);

    std::vector<AQPair> AQVector;

    std::vector<float> q_values(_shape.output);
    for(auto i=0;i<stateBatch.size();i++)
    {
        float max_q=0.0;
        uint32_t max_idx=0;
        for(auto j=0;j<_shape.output;j++)
        {
            float q = _q_values_blob->data_at(i, j, 0, 0);
            q_values[j]=q;
            //assert(!std::isnan(q));
            if(q>max_q)
            {
                max_idx=j;
                max_q=q;
            }
        }
        //if(stateBatch.size()==1) std::cout<<" Q"<<VectorToStr(q_values)<<std::endl;
        AQPair pair(max_idx,max_q);
        AQVector.push_back(pair);

    }

    return AQVector;

}


void DQN::AddTransition(const CDNN::DQN::Transition &transition)
{
    if(_replay_memory.size()==_replay_memory_capacity)
        _replay_memory.pop_front();
    _replay_memory.push_back(transition);
}

float DQN::Update()
{

    //std::cout << "iteration: " << _current_iter++ << std::endl;
    _current_iter++;
    //参与update的回放记录向量
    std::vector<int> TransSelectedVector;
    TransSelectedVector.reserve(_shape.batch);


    //随机选取回放记录库中的回放记录
    for (auto i = 0; i < _shape.batch;++i)
    {
        const auto random_transition_idx =
                std::uniform_int_distribution<int>(0, _replay_memory.size() - 1)(
                        _random_engine);
        TransSelectedVector.push_back(random_transition_idx);
    }


    //构造下一步的输入 其实S' 这里就得到 S' batches
    std::vector<DataUnitSptr> target_state_batch;
    for (const auto idx : TransSelectedVector)
    {
        const auto& transition = _replay_memory[idx]; //获得对应的回放记录
        auto target_state=std::get<3>(transition); // 获得 S'
        target_state_batch.push_back(target_state);
    }

    //获得S' 输入是的最大 Q(S',a')
    std::vector<AQPair> actions_and_values
            =SelectActionGreedilyBatch(target_state_batch);


    //构建训练网的输入输出来更新网络参数
    DataUnitSptr state_batch= DataUnitSptr(new DataUnit<float> (_in_size_batch,"state_batch"));
    DataUnitSptr target_input= DataUnitSptr(new DataUnit<float> (_shape.batch*_shape.output,"target_input"));
    DataUnitSptr filter_input= DataUnitSptr(new DataUnit<float> (_shape.batch*_shape.output,"filter_input"));

    target_input->FillZero();
    filter_input->FillZero();

    auto target_value_idx = 0;


    for (auto i = 0; i < _shape.batch; ++i) {
        auto& transition = _replay_memory[TransSelectedVector[i]]; //获得回放记录条目 s a r s'
        auto action = std::get<1>(transition); //获取 a

        assert(static_cast<int>(action) < _shape.output);
        auto reward = std::get<2>(transition); //获取 r
        //assert(reward >= -1.0 && reward <= 1.0);

        //完成 Q(s,a)=r+gamma*Q(s',a')
        auto target = reward + _gamma * actions_and_values[target_value_idx++].second;


        assert(!std::isnan(target));


        (target_input->begin())[i*_shape.output+action]=target;
        (filter_input->begin())[i*_shape.output+action]=1;


        //构造网络输入
        auto& state = std::get<0>(transition);

        state_batch->Copy(state->begin(),i*_in_size,_in_size);

    }

    //进行一次网络训练
    InputDataIntoLayers(state_batch, target_input, filter_input);
    /*
    state_batch->Show(200);
    std::cout<<std::endl;
    target_input->Show(10);
    std::cout<<std::endl;
    filter_input->Show(10);
    std::cout<<std::endl;*/
    _solver->Step(1);
   // std::cout<<"loss="<<_loss->data_at(0, 0, 0, 0)<<std::endl;
    return _loss->data_at(0, 0, 0, 0);

}

void DQN::InputDataIntoLayers(
        const DataUnitSptr& state_data,
        const DataUnitSptr& target_data,
        const DataUnitSptr& filter_data)
{
    _state_input_layer->Reset(
            state_data->begin(),
            _dummy_input_data->begin(),
            _shape.batch);
    _target_input_layer->Reset(
            target_data->begin(),
            _dummy_input_data->begin(),
            _shape.batch);
    _filter_input_layer->Reset(
            filter_data->begin(),
            _dummy_input_data->begin(),
            _shape.batch);

}



