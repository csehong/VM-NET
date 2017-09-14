from sys import argv  # allows user to specify input and output directories
import os  # help with file handling
import librosa_feature

SR = 12000
N_FFT = 512
HOP_LEN = 256
DURA = 29.12

def check_file_exists(directory, filename, extension):
    path = directory + "/" + filename + extension
    return os.path.isfile(path)


def main(indir, ext, outdir):
    try:
        # check specified folders exist
        if not os.path.exists(indir):
            exit("Error: Input directory \'" + indir + "\' does not exist. (try prepending './')")
        if not os.path.exists(outdir):
            exit("Error: Output directory \'" + outdir + "\' does not exist.")
        if not os.access(outdir, os.W_OK):
            exit("Error: Output directory \'" + outdir + "\' is not writeable.")

        print "[%s/*.mp3] --> [%s/feature]" % (indir, outdir)
        files = []  # files for exporting

        # get a list of all convertible files in the input directory
        filelist = [f for f in os.listdir(indir) if f.endswith(ext)]
        for path in filelist:
            basename = os.path.basename(path)
            filename = os.path.splitext(basename)[0]
            files.append(filename)
        # remove files that have already been outputted from the list
        files[:] = [f for f in files if not check_file_exists(outdir + "/pickle", f, ".pkl")]
        files[:] = [f for f in files if not check_file_exists(outdir + "/matlab", f, ".mat")]
    except OSError as e:
        exit(e)

    if len(files) == 0:
        exit("Could not find any files to convert that have not already been converted.")

    # convert all unconverted files
    for filename in files:
        librosa_feature.feature_extraction(indir, filename, ext, outdir, SR, N_FFT, HOP_LEN, DURA)


# # set the default directories and try to get input directories
# args = [".", "."]
# for i in range(1, min(len(argv), 3)):
#     args[i - 1] = argv[i]
#
# # if only input directory is set, make the output directory the same
# if len(argv) == 2:
#     args[1] = args[0]

ext = '.mp3'
indir = '/media/mass/D/wbim/project/youtube-8m_general/audio'
outdir = '/media/mass/D/wbim/project/youtube-8m_general/ge_audio_feature'

main(indir, ext, outdir)
