#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
M32=Qwen/Qwen3-VL-32B-Instruct
M8=Qwen/Qwen3-VL-8B-Instruct
L=logs/qwen
node() {
    local name="$1"; shift
    "$@" > "$L/relaunch_$name.out" 2>&1
}

a11_03() {
    bash runs/qwen/restart_server.sh 6787475 8,0 $L/serve_a11-03_32b_128k.out "" \
        $M32 -tp 4 --data-parallel-size 2 --max-model-len 131072 || return
    bash runs/qwen/launch_step.sh 6787475 8 $L/cur_32b_zelda_r2.out bash runs/qwen/lane_cur.sh $M32 legend_of_zelda_links_awakening legend_of_zelda_links_awakening,legend_of_zelda_the_oracle_of_seasons
    bash runs/qwen/launch_step.sh 6787475 8 $L/cur_32b_pokemon_r2.out bash runs/qwen/lane_cur.sh $M32 pokemon_red pokemon_red,pokemon_crystal
    bash runs/qwen/launch_step.sh 6787475 8 $L/cur_32b_deja_vu_r2.out bash runs/qwen/lane_cur.sh $M32 deja_vu_1 deja_vu_1,deja_vu_2
}

c04_06() {
    bash runs/qwen/restart_server.sh 6787491 6,0 $L/serve_c04-06_32b_128k.out "" \
        $M32 -tp 1 --max-model-len 131072 || return
    bash runs/qwen/launch_step.sh 6787491 8 $L/cur_32b_sword_r2.out bash runs/qwen/lane_cur.sh $M32 sword_of_hope_1 sword_of_hope_1,sword_of_hope_2
    bash runs/qwen/launch_step.sh 6787491 8 $L/cur_32b_bomberman_r2.out bash runs/qwen/lane_cur.sh $M32 bomberman_quest bomberman_quest,bomberman_pocket
}

c06_03() {
    bash runs/qwen/restart_server.sh 6787478 1,7,8,0 $L/serve_c06-03_8b_128k.out "" \
        $M8 -tp 1 --data-parallel-size 6 --max-model-len 131072 || return
    bash runs/qwen/launch_step.sh 6787478 8 $L/lane_8b_deja_vu_1_r2.out bash runs/qwen/lane.sh $M8 deja_vu_1 deja_vu_1,deja_vu_2 bench,practice
    bash runs/qwen/launch_step.sh 6787478 8 $L/cur_8b_zelda_r2.out bash runs/qwen/lane_cur.sh $M8 legend_of_zelda_links_awakening legend_of_zelda_links_awakening,legend_of_zelda_the_oracle_of_seasons
    bash runs/qwen/launch_step.sh 6787478 8 $L/cur_8b_pokemon_r2.out bash runs/qwen/lane_cur.sh $M8 pokemon_red pokemon_red,pokemon_crystal
}

a11_02() {
    bash runs/qwen/restart_server.sh 6787492 0 $L/serve_a11-02_8b_128k.out 7 \
        $M8 -tp 1 --data-parallel-size 4 --max-model-len 131072 || return
    bash runs/qwen/launch_step.sh 6787492 8 $L/cur_8b_bomberman_r2.out bash runs/qwen/lane_cur.sh $M8 bomberman_quest bomberman_quest,bomberman_pocket
}

node a11-03 a11_03 &
node c04-06 c04_06 &
node c06-03 c06_03 &
node a11-02 a11_02 &
wait
