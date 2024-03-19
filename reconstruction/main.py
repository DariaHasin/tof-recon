from utils import *

if __name__ == "__main__":
    today_date, current_time = get_today_date_time()
    output_title = get_output_title(today_date)
    output_volume = os.path.join(RECON_OUTPUT, output_title)
    input_stacks = get_input_stacks()

    # ===========================NeSVoR========================== #
    cmd_arg = []
    cmd_arg_print = []
    cmd_arg.append("nesvor reconstruct ")
    cmd_arg.append(f'--input-stacks ' + input_stacks)
    cmd_arg.append('--output-volume ' + output_volume )
    cmd_arg.append('--simulated-slices ' + SIM_SLICES )
    # cmd_arg.append(f'--stack-masks ' + mask_stacks)
    cmd_arg_print.append('--output-resolution 0.4 ')
    cmd_arg_print.append('--registration none ')
    # cmd_arg_print.append('--n-levels-bias 4 ')
    # cmd_arg_print.append('--output-model ' + output_model)
    cmd_print = ' '.join(cmd_arg_print)
    cmd_arg.append(cmd_print)
    cmd = ' '.join(cmd_arg)
    print(cmd)
    # !{cmd}
    # ============================================================ #

    args = "\n".join(cmd_arg_print).replace('--', '')
    row_data = {'date': today_date,
                'subject': SUBJECT,
                'file_title': output_title,
                'n_ax': PLANE_REPITITION['ax'],
                'n_cor': PLANE_REPITITION['cor'],
                'n_sag': PLANE_REPITITION['sag'],
                'args': args}
    add_row_of_current_recon(row_data, today_date, current_time)