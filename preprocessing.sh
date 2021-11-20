#!/bin/bash

if [ $# -ne 2 ]; then
	echo "Usage: ./run.sh [audio-dir] [nj]"
	exit
fi

audio_dir=$1
nj=$2

function convert_audio() {
	for f in $(cat $1); do
		echo $f
		wav_path=$(echo $f | sed 's@.mp3@.wav@g')
		[ ! -f $wav_path ] || rm ${wav_path}*
		ffmpeg -i $f -acodec pcm_s16le -ar 16000 -ac 1 ${wav_path}.wav > /dev/null 2>&1
		sox ${wav_path}.wav $wav_path silence 1 0.1 1%
	done
}

mkdir -p tmp
rm -rf tmp/*

find $audio_dir -type f -name *mp3 > tmp/all

total=$(wc -l < tmp/all)
per_job=$(echo "($total-1) / $nj + 1" | bc)

split -d -l $per_job tmp/all tmp/part

for part in $(ls tmp/part*); do
	convert_audio $part > ${part}.log 2>&1 &
done

wait