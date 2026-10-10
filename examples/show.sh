while true; do 
	ssh t41 squeue; echo;
	sqlite3 output/egfr.sqlite  "select *
from (
    select iteration,
           count(*),
           count(docking_score) as num_dock,
           printf('%6.2f', min(docking_score)) as glide,
           count(gbsa_score) as num_gbsa,
           printf('%6.2f', min(gbsa_score)) as gbsa,
           min(modified_at)
    from compound
    group by iteration
    order by iteration desc
    limit 10
)
order by iteration;"; 
        date;
	sleep 5s; 
done 

