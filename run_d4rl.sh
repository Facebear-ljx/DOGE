for SEED in 150 100 50 0; do
for ENV in 'walker2d' 'halfcheetah' 'hopper'; do
#ENV='halfcheetah'
for DATASET in 'medium' 'medium-replay' 'medium-expert' 'random'; do
ENV_NAME=$ENV'-'$DATASET'-v2'
echo $ENV_NAME

ALPHA=7.5

if [[ "$ENV_NAME" =~ "hopper-medium-replay" ]]; then
ALPHA=17.5
fi

if [[ "$ENV_NAME" =~ "hopper-medium-v2" ]]; then
ALPHA=17.5
fi

if [[ "$ENV_NAME" =~ "random" ]]; then
ALPHA=17.5
fi

echo $ALPHA $ ENV_NAME
python train_distance_mujoco.py --env_name $ENV_NAME --alpha $ALPHA
done
done
done