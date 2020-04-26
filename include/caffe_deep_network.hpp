#ifndef CAFFE_DEEP_NETWORK_HPP
#define CAFFE_DEEP_NETWORK_HPP


#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <caffe/caffe.hpp>
#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>
#include <caffe/layers/memory_data_layer.hpp>


namespace CDNN  //caffe deep neural network
{

    template <typename dType>
    class DataUnit
    {
    public:
        typedef boost::shared_ptr<DataUnit> sptr;
    private:
        dType *_data;
        uint32_t _size;
        std::string _name;
    public:
        DataUnit(uint32_t size ,std::string name):_size(size),_name(name)
        {
            _data=new dType[size];
        }
        ~DataUnit()
        {
          //  std::cout<<"delete unit data "<<_name<<" "<<_size<<std::endl;
            delete (dType *) _data;
        }

        uint32_t size() {return _size;}
        dType *begin() {return _data;}
        void Copy(dType *source, uint32_t dest, uint32_t len)
        {
            assert(dest+len<=_size);
            memcpy(_data+dest,source,len*sizeof(dType));
        }

        double Sum()
        {
            double sum=0;
            for(auto i=0;i<_size;i++)
            {
                sum=sum+_data[i];
            }
            return sum;
        }

        void FillZero(){
            memset(_data,0,_size*sizeof(dType));
        }

        void FillRandom(uint32_t range)
        {
            for(auto i=0;i<_size;i++)
            {
                _data[i]=random() % range;
            }
        }

        void Show(uint32_t num_per_line)
        {
            std::cout<<_name<<","<<_size<<std::endl;
            for(auto i=0;i<_size;i++)
            {
                std::cout<<_data[i]<<" ";

                if( ((i+1) % num_per_line)==0) std::cout<<std::endl;

            }
            std::cout<<std::endl;
        }


    };

    using DataUnitSptr= DataUnit<float>::sptr;

    typedef struct Shape
    {
        uint32_t batch;
        uint32_t channels;
        uint32_t height;
        uint32_t width;
        uint32_t output;
    } Shape_t;

    class DQN
    {
    public:

        typedef boost::shared_ptr<DQN> sptr;
        //变量类型
        using Action = uint32_t;
        using Reward = float;
        using ActionVector = std::vector<Action>;
        using AQPair=std::pair<Action, float>;
        using StateBatch =std::vector<DataUnitSptr>;

        using Transition = std::tuple<DataUnitSptr, Action, Reward, DataUnitSptr>;

    private:

        const std::string _solver_param;
        const uint32_t _replay_memory_capacity;
        const double _gamma;
        uint32_t _current_iter;
        std::mt19937 _random_engine;
        uint32_t _in_size;
        uint32_t _in_size_batch;
        uint32_t _out_size;
        Shape_t _shape;





        //网络结构指针
        using SolverSp = boost::shared_ptr<caffe::Solver<float>>;
        using NetSp = boost::shared_ptr<caffe::Net<float>>;
        using BlobSp = boost::shared_ptr<caffe::Blob<float>>;
        using MemoryDataLayerSp = boost::shared_ptr<caffe::MemoryDataLayer<float>>;





        //网络数据接口
        SolverSp _solver;
        NetSp _net;
        BlobSp _q_values_blob;
        BlobSp _loss;
        MemoryDataLayerSp _state_input_layer;
        MemoryDataLayerSp _target_input_layer;
        MemoryDataLayerSp _filter_input_layer;
        DataUnitSptr  _dummy_input_data;

        std::deque<Transition> _replay_memory;



    public:


        DQN(const std::string & solver_param, const int replay_memory_capacity, const double gamma):
        _solver_param(solver_param),
        _replay_memory_capacity(replay_memory_capacity),
        _gamma(gamma),
        _current_iter(0),
        _random_engine(0){}

        //初始化
        void Initialize(const Shape &shape);

        //选择策略
        Action SelectAction(DataUnitSptr& state, double epsilon);

        //增加跳转记录
        void AddTransition(const Transition& transition);


        int memory_size() const { return _replay_memory.size(); }


        float Update();

        uint32_t Iteration(){return _current_iter;}

    private:
        AQPair SelectActionGreedily(DataUnitSptr& state);
        std::vector<AQPair> SelectActionGreedilyBatch(StateBatch& stateBatch);

        void InputDataIntoLayers(
                const DataUnitSptr& state_data,
                const DataUnitSptr& target_data,
                const DataUnitSptr& filter_data);


    };

}


#endif