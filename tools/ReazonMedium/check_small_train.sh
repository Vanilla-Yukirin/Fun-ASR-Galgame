SMALL_TRAIN_SCP="/root/autodl-tmp/ML/Reazon/small/train/manifests/train_reazon_temp.scp"
MEDIUM_AUDIO_DIR="/root/autodl-tmp/ML/Reazon/medium"
WORKDIR="/tmp/reazon_overlap_check1"

mkdir -p "$WORKDIR"

# small train scp -> basename 集合
awk '{n=split($2,a,"/"); f=a[n]; sub(/\.[^.]+$/, "", f); print f}' "$SMALL_TRAIN_SCP" \
  | sort -u > "$WORKDIR/small_train.base"

# medium 目录递归扫音频 -> basename 集合
find "$MEDIUM_AUDIO_DIR" -type f \( -iname '*.flac' -o -iname '*.wav' -o -iname '*.mp3' -o -iname '*.ogg' -o -iname '*.m4a' \) \
  | awk '{n=split($0,a,"/"); f=a[n]; sub(/\.[^.]+$/, "", f); print f}' \
  | sort -u > "$WORKDIR/medium.base"

# 交集
comm -12 "$WORKDIR/small_train.base" "$WORKDIR/medium.base" > "$WORKDIR/overlap_by_basename.txt"

echo "[SMALL base] $(wc -l < "$WORKDIR/small_train.base")"
echo "[MEDIUM base] $(wc -l < "$WORKDIR/medium.base")"
echo "[OVERLAP]    $(wc -l < "$WORKDIR/overlap_by_basename.txt")"
echo "---- overlap head 30 ----"
head -n 30 "$WORKDIR/overlap_by_basename.txt"