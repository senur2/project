module load conda
conda activate pytorch
2. Configuration du fichier de sortie
OUTPUT="resultats_gpu.txt"
DATASET="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
MODEL="mnasnet1_0"

# On crée l'entête du fichier CSV (On écrase l'ancien fichier avec '>')
# ATTENTION : Assurez-vous que l'entête correspond à ce que votre Python écrit !
echo "LoadingTime,ComputeTime,Cores,BatchSize" > $OUTPUT

# 3. La Boucle de Test (Batch Sizes)
# On teste : 16, 32, 64, 128
for BATCH in 16 32 64 128
do
    echo "=========================================================="
    echo "TESTING BATCH SIZE: $BATCH"
    echo "=========================================================="

    # --- Test A : 1 GPU ---
    echo "  > Running on 1 GPU..."
    # Note : On passe $BATCH en 3ème argument
    # Note : On passe "1" en dernier argument (pour dire 1 cœur/GPU utilisé)
    torchrun --nnodes=1 --nproc_per_node=1 demo_gpu.py \
        "$MODEL" \
        "$DATASET" \
        "$BATCH" \
        "$OUTPUT" \
        "1"

    # --- Test B : 2 GPUs (DDP) ---
    echo "  > Running on 2 GPUs..."
    # Note : On passe "2" en dernier argument
    torchrun --nnodes=1 --nproc_per_node=2 demo_gpu.py \
        "$MODEL" \
        "$DATASET" \
        "$BATCH" \
        "$OUTPUT" \
        "2"

    # Petite pause pour laisser le GPU vider sa mémoire et refroidir
    sleep 2
done

echo "=========================================================="
echo "Benchmark terminé. Voici les résultats :"
echo "=========================================================="
cat $OUTPUT