\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{storm-ai_report}[] % LuaLatex templace created for MIT ARCLab IA Challenge for Space Innovation 2025 (STORM-AI)

% Load standard LaTeX classes and packages
\LoadClass[12pt, a4paper]{article}
\RequirePackage{hyperref}
\RequirePackage{fontspec}
\hypersetup{
    pdfstartview=FitH,
    colorlinks=true,
    linkcolor=MITRed,
    citecolor=MITRed,
    urlcolor=MITRed
}
\RequirePackage[top=1cm, bottom=2cm, left=1cm, right=1cm]{geometry}
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{multicol}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{fontawesome}
\RequirePackage{tikz}
\RequirePackage[style=ieee, backend=biber, citestyle=numeric-comp]{biblatex}
\RequirePackage{tcolorbox}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{etoolbox} 
\RequirePackage{environ} 
\RequirePackage{ifthen}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}

% --- MIT Font Configuration ---
\setmainfont{Times New Roman}
\newfontfamily\MITSans{Arial}
\newcommand{\useSerifFont}{\rmfamily\setmainfont{Times New Roman}}
\newcommand{\useSansFont}{\sffamily\MITSans}

% --- MIT Brand Colors ---
\definecolor{MITRed}{HTML}{A31F34}
\definecolor{MITGray}{HTML}{8A8B8C}
\definecolor{MITLightGray}{HTML}{C2C0BF}
\definecolor{MITBlack}{HTML}{000000}
\definecolor{MITDateRed}{HTML}{FF1423}
\definecolor{darkgray}{RGB}{112,111,111}

% Legacy Pantone colors (not used)
\definecolor{pantone286}{RGB}{0,51,160}
\definecolor{pantone285}{RGB}{0,118,194}
\definecolor{pantone2995}{RGB}{0,174,239}
\definecolor{pantone877C}{RGB}{200,200,200}

% Bibliography font
\renewcommand*{\bibfont}{\fontsize{10}{12}\selectfont}

% Caption setup
\DeclareCaptionFormat{custom}{%
    \fontsize{10}{12}\textbf{#1#2}#3
}
\captionsetup{format=custom, font=footnotesize, width=0.6\columnwidth}

% Top bar
\newcommand{\topbar}[1]{%
  \noindent\colorbox{MITBlack}{%
    \parbox{\dimexpr\textwidth-2\fboxsep\relax}{%
      \MITSans\centering\color{white}\fontsize{14}{16}\selectfont%
      \vspace{0.2em}#1\vspace{0.2em}%
    }%
  }%
}

% Header box
\newcommand{\headerbox}[1]{%
    \noindent
    \colorbox{MITRed}{%
        \parbox{\dimexpr\textwidth-2\fboxsep\relax}{%
            \vspace{0.8em}
            \begin{minipage}[c]{0.2\textwidth} 
                \includegraphics[width=\linewidth]{images/logos/clr_arclab_logo.png}
            \end{minipage}
            \hfill
            \begin{minipage}[c]{0.5\textwidth} 
                \centering
                \linespread{1.8}
                \MITSans\textbf{\textcolor{white}{\fontsize{22pt}{26pt}\selectfont#1}}
            \end{minipage}
            \hfill
            \begin{minipage}[c]{0.2\textwidth} 
                \includegraphics[width=\linewidth]{images/logos/clr_ai_accelerator_logo.png}
            \end{minipage}
            \vspace{0.8em}
        }%
    }%
}

% Authors logic
\newcounter{participants}
\newcommand{\speaker}[2]{%
  \stepcounter{participants}%
  \expandafter\def\csname speakername\arabic{participants}\endcsname{#1}%
  \expandafter\def\csname speakeremail\arabic{participants}\endcsname{#2}%
}
\NewEnviron{participantbox}{%
  % Collect user-supplied \speakernameN / \speakeremailN macros
  \setcounter{participants}{0}
  \BODY

  % ---- layout parameters (adjust as needed) ----
  \def\nodewidth{7.8}        % wider chip
  \def\nodeheight{3}
  \def\nodespacing{1}      % horizontal gap
  \def\nodespacingy{0.5}   % vertical gap
  % ---------------------------------------------

  \noindent\colorbox{MITLightGray}{%
    \parbox{\dimexpr\textwidth-2\fboxsep\relax}{%
      \vspace{0.2em}%
      \begin{center}
        \begin{tikzpicture}
          % -- batch & index counters --
          \newcounter{batchnum}\setcounter{batchnum}{0}
          \newcounter{totalbatches}
          \setcounter{totalbatches}{\value{participants}}
          \addtocounter{totalbatches}{1}  % ceil((participants)/2)
          \divide\c@totalbatches by 2
          \newcounter{speakerindex}\setcounter{speakerindex}{1}
          \newcounter{batchend}\newcounter{batchsize}
          \newcounter{i}\newcounter{speakerindexi}

          % -- main loop over rows --
          \whiledo{\value{batchnum} < \value{totalbatches}}{%
            % Row end index = current + 1  (2 per row)
            \setcounter{batchend}{\value{speakerindex}}
            \addtocounter{batchend}{1}
            \ifthenelse{\value{batchend} > \value{participants}}{%
              \setcounter{batchend}{\value{participants}}}{}

            % batchsize = (# chips this row)
            \setcounter{batchsize}{\value{batchend}}
            \addtocounter{batchsize}{-\value{speakerindex}}
            \addtocounter{batchsize}{1}

            % y-coordinate for this row
            \pgfmathsetmacro{\ycoord}{-(\nodeheight+\nodespacingy)*\value{batchnum}}

            % ---- loop within the row ----
            \setcounter{i}{0}
            \whiledo{\value{i} < \value{batchsize}}{%
              \setcounter{speakerindexi}{\value{speakerindex}}
              \addtocounter{speakerindexi}{\value{i}}

              % x-coordinate: centred for 1, symmetric for 2
              \ifthenelse{\value{batchsize}=1}{%
                \def\speakerx{0}%
              }{%
                \ifthenelse{\value{i}=0}{%
                  \pgfmathsetmacro{\speakerx}{-(\nodewidth/2+\nodespacing/2)}%
                }{%
                  \pgfmathsetmacro{\speakerx}{(\nodewidth/2+\nodespacing/2)}%
                }%
              }%

              % Fetch name / e-mail macros
              \edef\currentspeakername{\csname speakername\arabic{speakerindexi}\endcsname}
              \edef\currentspeakeremail{\csname speakeremail\arabic{speakerindexi}\endcsname}

              % Draw the chip
              \node[draw=white, text=white, rounded corners=12pt,
                    fill=darkgray, text width=\nodewidth cm,
                    align=center, minimum height=\nodeheight cm]
                at (\speakerx cm,\ycoord cm) {%
                  \MITSans\textbf{\currentspeakername}\\[0.15em]
                  \scriptsize\itshape\currentspeakeremail};

              \stepcounter{i}
            }

            % -- advance to next row --
            \setcounter{speakerindex}{\value{batchend}}
            \addtocounter{speakerindex}{1}
            \stepcounter{batchnum}
          }
        \end{tikzpicture}
      \end{center}
    }%
  }%
}
% Author date box
\newcommand{\authordate}[2]{%
  \noindent
  \begin{tcolorbox}[
    colback=MITBlack,
    colframe=MITBlack,
    width=\textwidth,
    boxrule=0pt,
    left=1em,
    right=1em,
    top=0.5em,
    bottom=0.5em,
    sharp corners,
    before skip=0.2em,
    after skip=0.2em
  ]
  \MITSans\small\color{white}\textbf{}#1\hfill\textbf{Date: }#2%
  \end{tcolorbox}
}

% Table of contents
\makeatletter
\renewcommand{\tableofcontents}{%
  \begingroup
  \hypersetup{linkcolor=MITBlack}%
  \begin{tcolorbox}[
    colback=MITLightGray!10,
    colframe=MITGray,
    width=\columnwidth,
    boxrule=0.5pt,
    left=1em,
    right=1em,
    top=1em,
    bottom=1em,
    before skip=1em,
    after skip=1em
  ]
  \MITSans\textbf{\large\textcolor{MITRed}{Table of Contents}} \\[-1.5em]
  
  {\small\raggedright\@starttoc{toc}}%
  \end{tcolorbox}%
  \endgroup
}
\makeatother

% Abstract box
\newcommand{\abstractbox}[2]{%
  \begin{tcolorbox}[
    colback=MITLightGray!10,
    colframe=MITLightGray,
    width=\columnwidth,
    boxrule=0.5pt,
    left=1em,
    right=1em,
    top=1em,
    bottom=1em,
    before skip=1em,
    after skip=1em
  ]
  \noindent\textit{\MITSans\large\textcolor{darkgray}{#1}} \\
  #2
  \end{tcolorbox}
}

% Section titles
\titleformat{\section}{\MITSans\Large\bfseries\color{MITGray}}{}{0em}{}[{\color{MITRed}\titlerule[1.5pt]}]
\titlespacing*{\section}{0pt}{20pt}{12pt}
\titleformat{\subsection}{\MITSans\large\bfseries\color{MITGray}}{}{0em}{}
\titlespacing*{\subsection}{0pt}{15pt}{8pt}

% Footer
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\endinput