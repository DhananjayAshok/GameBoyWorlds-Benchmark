# Deletes all replay buffers, models, curiosity buffers, tmp data and per-game emulator
# sessions from the storage directory, and calls clean_cache.sh. Guarded by an intentional
# exit at the top; remove that line to actually run this.
echo "Uncomment to run"; exit 1
source scripts/core/utils.sh
bash scripts/core/clean_cache.sh
currpath=$(pwd)
cd $storage_dir
echo "Cleaning up all data folders in $storage_dir..."
rm -rf curiosity_buffers models replay_buffers tmp
cd $currpath
source GameBoyWorlds/configs/config.env
echo "Clearing sessions in GameBoyWorlds..."
games=("pokemon_red" "pokemon_crystal" "pokemon_fools_gold" "pokemon_starbeasts" "sword_of_hope_1" "sword_of_hope_2" "deja_vu_1" "deja_vu_2" "harvest_moon_1" "harvest_moon_2" "harvest_moon_3" "harrypotter_philosophersstone" "legend_of_zelda_links_awakening" "legend_of_zelda_the_oracle_of_seasons" "bomberman_pocket" "bomberman_quest" "bomberman_max")
for game in "${games[@]}"; do
    echo "Cleaning sessions for game: $game"
    rm -rf $storage_dir/sessions/$game/${game}_*
done
source scripts/core/utils.sh