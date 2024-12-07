To create a new task that re-uses this orchestration's scp settings:
1. Copy launch.sh and test.dstack.yml into your new task
2. Update the name of test.dstack.yml to a suitable name
3. Update the forked launch.sh to match the newly named dstack.yml
4. Add a new script that serve's as the tasks main and update RUN_COMMAND in the
   dstack.yml to call this new script.
