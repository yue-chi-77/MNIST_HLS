# Rebuild the v1 KD240 design with a different PL clock constraint.
#
# Experiment A: the shipped bitstream was constrained at 100 MHz, so Vivado
# stopped optimising as soon as it hit 9.0 ns. Re-run the *identical* design
# at 250 MHz and see what WNS the tool can actually reach when it has to try.
#
# Usage: vivado -mode batch -source build_impl.tcl -tclargs <bd_tcl> <out_dir> [strategy]

set bd_tcl   [lindex $argv 0]
set out_dir  [lindex $argv 1]
set strategy [lindex $argv 2]

set part       xck24-ubva530-2LV-c
set ip_repo    [file normalize [file join [file dirname [info script]] ip_repo]]
set jobs       8

file mkdir $out_dir
create_project -force impl $out_dir -part $part
set_property BOARD_PART xilinx.com:kd240_som:part0:1.1 [current_project]
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog -rebuild

source $bd_tcl

set bd [get_files design_1.bd]
make_wrapper -files $bd -top -import
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

if {$strategy ne ""} {
    set_property strategy $strategy [get_runs impl_1]
}

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    puts "ERROR: synthesis failed"
    exit 1
}

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    puts "ERROR: implementation failed"
    exit 1
}

open_run impl_1

# The two numbers this experiment exists to produce.
set wns [get_property SLACK [get_timing_paths -delay_type max]]
set period [get_property PERIOD [get_clocks clk_pl_0]]
puts "RESULT_PERIOD_NS $period"
puts "RESULT_WNS_NS $wns"
puts "RESULT_FMAX_MHZ [expr {1000.0 / ($period - $wns)}]"

report_timing_summary -file $out_dir/timing_summary.rpt
report_timing -delay_type max -max_paths 10 -nworst 1 -file $out_dir/timing_worst.rpt
report_utilization -file $out_dir/utilization.rpt
exit 0
