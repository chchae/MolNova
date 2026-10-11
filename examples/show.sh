while true; do 
	ssh t41 squeue; echo;
	sqlite3 -cmd ".mode box --quote off" output/egfr.sqlite  "select *
from (
    select iteration,
           count(*) as count,
           count(docking_score) as n_dock,
           printf('%6.2f', min(docking_score)) as glide,
           count(gbsa_score) as n_gbsa,
           printf('%6.2f', min(gbsa_score)) as gbsa,
           strftime('%m-%d %H:%M', MIN(modified_at), '+9 hours') AS modified
    from compound
    group by iteration
    order by iteration desc
    limit 10
)
order by iteration;"; 
        date;
	sleep 5s; 
done 

