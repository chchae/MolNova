num_limit=10;
while true; do 
        echo "--------------------------------------------------------------"
        date;
	ssh -n t41 squeue; echo;
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
    limit $num_limit
)
order by iteration;"; 

        if IFS= read -r -s -n 1 -t 1 key; then
            case "$key" in
                " ")
                    num_limit=10000;
                    continue;;
                "")
                    num_limit=10000;
                    continue;;
            esac
        else
            num_limit=10;
	    sleep 5s;
        fi

        echo; echo;

done 

