# Set to continue

find /projects/magstructsADSP/trhone/RuTMX/production/Ru_prod-db/data -name INCAR -exec sed -i 's/ISTART = 0/ISTART = 1/g' {} \;

find /projects/magstructsADSP/trhone/RuTMX/production/Ru_prod-db/data -name INCAR -exec sed -i 's/ICHARG = 2/ICHARG = 1/g' {} \;

find /projects/magstructsADSP/trhone/RuTMX/production/Ru_prod-db/data -name INCAR -exec sed -i 's/LWAVE = .FALSE./LWAVE = .TRUE./g' {} \;

copy (or diff and manually update) post.py from my source

diff parse_outcar.py -- in case you have any changes from mine

balsam ls apps --verbose # make sure that post.py is the correct one!

# python: reset the initial jobs
failed_initial = BalsamJob.objects.filter(state="FAILED", name="initial")
for job in failed_initial:
   job.data["retry_count"] = 0
   job.save()
   if "reached maximum retry" in job.state_history:
       job.update_state("RESTART_READY") # rerun jobs that have timed out 4 times
   else:
       job.update_state("RUN_ERROR") # re-mark error to handle errors with new post.py


## Then after you have fixed the failed initial calcs this way, you can reset the failed children like this:
from balsam.launcher.dag import BalsamJob
failed_children = BalsamJob.objects.filter(state_history__contains="One or more parent jobs failed")
BalsamJob.batch_update_state(failed_children, "AWAITING_PARENTS", "reset parents, try again")

## This way the failed initial calcs and their children can run again

# look for failed jobs which hit max retry limit and restart..
from balsam.launcher.dag import BalsamJob
restart_jobs = BalsamJob.objects.filter(state="FAILED", state_history__contains="reached maximum retry")
for job in restart_jobs:
   job.data["retry_count"] = 0
   job.save()
   job.update_state("RESTART_READY", "reset retry counter to 0; keep going")
