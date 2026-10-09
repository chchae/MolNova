while true; do 
	ssh t41 squeue; echo;
	sqlite3 output/egfr.sqlite  "select *
from (
    select iteration,
           count(*),
           count(docking_score),
           count(gbsa_score),
           printf('%6.2f', min(docking_score)) as glide,
           printf('%6.2f', min(gbsa_score)) as gbsa
    from compound
    group by iteration
    order by iteration desc
    limit 10
)
order by iteration;"; 
	sleep 5s; 
done 

