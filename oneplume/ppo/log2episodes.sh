cd /Users/aaravsinha/manyplume/oneplume/ppo/

# Set the base directory to where your model evaluation data is stored
BASEDIR="/Users/aaravsinha/manyplume/oneplume/ppo/trained_models/ExptMemory20250720/"

# Find all pickle files generated from your evaluation
LOGFILES=$(find ${BASEDIR} -name "*.pkl")
echo "Found pickle files:"
echo $LOGFILES

MAXJOBS=3
for LOGFILE in $LOGFILES; do
  while (( $(jobs -p | wc -l) >= $MAXJOBS )); do sleep 1; done
  python -u log2episodes.py --logfile $LOGFILE >> /dev/null 2>&1 &
done

# Wait for all jobs to complete
wait
echo "Episode extraction complete!"

# Check what was generated
find ${BASEDIR} -name "*_episodes.csv"
find ${BASEDIR} -name "*_start_end.png"