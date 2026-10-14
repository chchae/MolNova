num_limit=10;
while true; do 
        echo "----------------------------------------------------------------------------"
        date;
	ssh -n t41 squeue; echo;
    if [ -f output/egfr.sqlite ] &&
           [ "$(sqlite3 output/egfr.sqlite \
               "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='compound';" \
               2>/dev/null)" = "1" ]; then

sqlite3 -cmd ".mode box --quote off" output/egfr.sqlite  "select *
from (
    select iteration as cycle,
           count(*) as count,
           count(docking_score) as n_glid,
           printf('%6.2f', min(docking_score)) as glide,
           count(amber_gbsa_score) as n_ambe,
           printf('%6.2f', min(amber_gbsa_score)) as amber,
           count(gbsa_score) as n_prim,
           printf('%6.2f', min(gbsa_score)) as prime,
           strftime('%H:%M', MIN(modified_at), '+9 hours') AS modified
    from compound
    group by cycle
    order by cycle desc
    limit $num_limit
)
order by cycle;";

    fi

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

