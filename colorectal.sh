while [ "$1" != "" ];
do
   case $1 in
    -cp | --checkpoint_file )
        shift
        CHECKPOINT_FILE="$1"
        echo "Checkpoint File is : $CHECKPOINT_FILE"
        ;;
    --OUTPUT_DIR )
        shift
        OUTPUT_DIR="$1"
        echo "Output Directory is : $OUTPUT_DIR"
        ;;
    --DATASET_DIR )
        shift
        DATASET_DIR="$1"
        echo "Dataset Directory is : $DATASET_DIR"
        ;;
    --PRECISION)
        shift
        PRECISION="$1"
        if [ $PRECISION = "Mixed_Precision" ]
           then
           export TF_ENABLE_AUTO_MIXED_PRECISION=1
           echo "Mixed Precison set by user"
        fi
        ;;
    --inference )
        INFERENCE=1
        echo "Inference option is : $INFERENCE"
        ;;
    --PLATFORM )
        shift
        PLATFORM="$1"
        ;;
    --BATCH_SIZE )
        shift
        BATCH_SIZE="$1"
        ;;
    --NUM_EPOCHS )
        shift
        NUM_EPOCHS="$1"
        ;;
    -h | --help )
         echo "Usage: colorectal.sh [OPTIONS]"
         echo "OPTION includes:"
         echo "   --PRECISION - whether to use FP32 precision or Mixed_Precision Options : [FP32(default),Mixed_Precision]"
         echo "   --platform - To optimize for SPR : [None(default)SPR]"
         echo "   --inference - whether to run only inference"
         echo "   --cp  - Specify checkpoint directory for inference"
         echo "   --OUTPUT_DIR  - Specify output Directory need to be saved"
         echo "   --DATASET_DIR  - Specify dataset Directory"
         echo "   --BATCH_SIZE - Batch Size for training[32(default)]"
         echo "   --NUM_EPOCHS  - Num epochs for training[100(default)]"
         echo "   -h | --help - displays this message"
         exit
      ;;
    * )
        echo "Invalid option: $1"
        echo "Usage: colorectal.sh [OPTIONS]"
        echo "OPTION includes:"
        echo "   --PRECISION - whether to use Mixed_Precision or FP32 precision : [FP32(default),Mixed_Precision]"
        echo "   --platform - To optimize for SPR : [None(default),SPR]"
        echo "   --inference - whether to run only inference"
        echo "   --cp  - Specify checkpoint directory for inference"
        echo "   --OUTPUT_DIR  - Specify output Directory need to be saved"
        echo "   --DATASET_DIR  - Specify dataset Directory"
        echo "   --BATCH_SIZE - Batch Size for training[32(default)]"
        echo "   --NUM_EPOCHS  - Num epochs for training[100(default)]"
        exit
       ;;
  esac
  shift
done
if [ ! -d $OUTPUT_DIR ]
  then
  mkdir "$OUTPUT_DIR"
fi
if [ -z "$PRECISION" ]; then
    echo "Precision is not set"
    PRECISION="FP32"
fi
if [[ -z "${INFERENCE}" ]]; then
    echo "INFERENCE Default value is zero"
    INFERENCE=0
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "OUTPUT Directory default value is"
    OUTPUT_DIR="logs/fit/"
fi
if [[ -z "${DATASET_DIR}" ]]; then
    echo "Dataset Directory default value is"
    DATASET_DIR="datasets/"
fi
if [[ -z "${BATCH_SIZE}" ]]; then
    echo "Batch Size setting default values as 32"
    BATCH_SIZE=32
fi
if [[ -z "${NUM_EPOCHS}" ]]; then
    echo "Num epochs setting default value of 100"
    NUM_EPOCHS=100
fi
LOG_FILE=$OUTPUT_DIR/result.txt
if [ $PLATFORM = "SPR" ]
 then
 echo "Platform is SPR"
 export TF_ENABLE_ONEDNN_OPTS=1
 cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
 thread_per_core=$(lscpu |grep 'Thread(s) per core:' |sed 's/[^0-9]//g')
 export intra_op_parallelism_threads=$cores_per_socket
 export inter_op_parallelism_threads=1
 export KMP_BLOCKTIME=1
 export KMP_SETTINGS=1
 export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
 export OMP_NUM_THREADS=$cores_per_socket
 start_index=0
 end_index=$(($cores_per_socket - 1))
 ht_start_index=$(($cores_per_socket * 2))
 ht_end_index=$(($ht_start_index + $end_index))
 if [[ $thread_per_core = 1 ]]; then
     echo "HT OFF"
     range="$start_index-$end_index"
 else
     range="$start_index-$end_index,$ht_start_index-$ht_end_index"
 fi
 numactl -C $range -m 0 python colorectal_tflearn.py --precision=$PRECISION --inference "$INFERENCE" \
  --cp "$CHECKPOINT_FILE" --platform=$PLATFORM --OUTPUT_DIR=$OUTPUT_DIR --DATASET_DIR=$DATASET_DIR --BATCH_SIZE=$BATCH_SIZE --NUM_EPOCHS=$NUM_EPOCHS > $LOG_FILE
 else
  python colorectal_tflearn.py --precision=$PRECISION --inference "$INFERENCE" --cp "$CHECKPOINT_FILE" --platform="$PLATFORM" \
  --OUTPUT_DIR=$OUTPUT_DIR --DATASET_DIR=$DATASET_DIR --BATCH_SIZE=$BATCH_SIZE --NUM_EPOCHS=$NUM_EPOCHS > $LOG_FILE
fi