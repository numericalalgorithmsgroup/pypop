\LoadClass[11pt, a4paper, twoside]{article}
% , parskip=half*
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pop_report}[2015/11/11 POP Project class]

\usepackage{hyperref}
\usepackage{xcolor}
\hypersetup{
    colorlinks,
    linkcolor={black!50!black},
    citecolor={blue!50!black},
    urlcolor={blue!80!black}
}

\RequirePackage{color}
\RequirePackage{colortbl}
\RequirePackage{fancyhdr}
\RequirePackage{fullpage}
\RequirePackage{graphicx}
\RequirePackage{tabularx}
\RequirePackage[small]{titlesec}
\RequirePackage{xspace}
\RequirePackage{hyphenat}
\RequirePackage{lmodern}
\RequirePackage[english]{babel}

\graphicspath{{./graphics}}

% Title commands
\newcommand{\reprefnum}[1]{\def\@reprefnum{#1}}
\newcommand{\application}[1]{\def\@application{#1}}
\newcommand{\servicelevel}[1]{\def\@servicelevel{#1}}
\newcommand{\contributors}[1]{\def\@contributors{#1}}
\newcommand{\keywords}[1]{\def\@keywords{#1}}
\title{POP Ref.No.~\@reprefnum}

\renewcommand\maketitle{
    \thispagestyle{empty}
    \begin{center}
    \begin{figure}[h]
    \centering
    \hspace{1cm}\Oldincludegraphics[width=12cm]{pop-logo}
    \end{figure}
    \end{center}

    \vspace{1cm}

    \begin{center}
    \Large\bf\color[rgb]{0.44, 0.71, 0.04}
    \textsf{\@title}
    \end{center}

    \vspace{1.5cm}

    \begin{flushleft}
    \Large\bf\color[rgb]{0.44, 0.71, 0.04}
    \textsf{Document Information}
    \end{flushleft}
    \vspace{-2em}
    \begin{center}
    \begin{tabular}{|l|p{0.7\textwidth}|}
    \hline
    \color[rgb]{0.14, 0.25, 0.38}Reference Number & \@reprefnum \\ \hline
    \color[rgb]{0.14, 0.25, 0.38}Author & \@author \\ \hline
    \color[rgb]{0.14, 0.25, 0.38}Contributor(s) & \@contributors \\ \hline
    \color[rgb]{0.14, 0.25, 0.38}Date & \@date \\ \hline
    \end{tabular}
    \end{center}

    \vfill

    {\footnotesize
    \em
    \textbf{Notices:}
    The research leading to these results has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 676553.
    }
   \\
   \\
  {\begin{tabular}{p{2cm} p{12cm}}
    \vspace{0pt}
    \Oldincludegraphics[width=2cm]{EU-logo}
    & \footnotesize
    \vspace{0pt}
     \copyright 2015 POP Consortium Partners. All rights reserved.\\
    \end{tabular}
    }
   \clearpage
}

\newcommand{\frontmatter}{
    \vspace*{-0.9cm}
    \tableofcontents
    \clearpage
}


\newenvironment{changelog}{
    \begin{flushleft}
    \LARGE\bf\color[rgb]{0.14, 0.25, 0.38}
    \vspace*{0.5cm}
    \textsf{Change Log}
    \end{flushleft}

    \begin{center}
    \begin{tabular}{|l|l|l|}
    \hline
    \rule[-0.3cm]{0mm}{0.8cm} \centering \bf \textsf{Version} & \centering \bf \textsf{Author} & \centering \bf \textsf{Description of Change} \tabularnewline
    } {
    \hline
    \end{tabular}
    \end{center}
    }

\newcommand{\change}[3]{
    \hline
    \rule[-0.3cm]{0mm}{0.8cm} \centering v#1 & #2 & #3 \\
}

\titleformat{\section}[hang]{\huge\bfseries\sffamily\color[rgb]{0.14, 0.25, 0.38}}{\thesection}{1em}{}
\titleformat{\subsection}[hang]{\Large\bfseries}{\thesubsection}{1em}{}

\newcommand{\fixme}[1]{\textcolor{red}{\textit{FIXME $->$ #1 $<-$}}}
\newcommand{\todo}[1]{\textcolor{red}{\textbf{TODO: #1}}}
\newcommand{\eg}{e.g.,\xspace}
\newcommand{\ie}{i.e.,\xspace}
\newcommand{\etc}{etc.\@\xspace}
\newcommand{\etal}{et.\ al.\xspace}
\newcommand{\vs}{vs.\xspace}
\newcommand{\IO}{I\slash O\xspace}

\newcommand{\slogo}{\Oldincludegraphics[width=2cm]{pop-logo}}

\AtBeginDocument{
    \headheight=30pt
    \pagestyle{fancy}
    \fancyhf{}
    \headsep 2ex
    \footskip 4ex
    \rhead[\slogo]{\slogo}
    \chead[\rightmark]{\rightmark}
    \lhead[ {\scriptsize \@title}]{{\scriptsize \@title}}
    \cfoot{\thepage}
    \renewcommand{\sectionmark}[1]{\markboth{#1}{}}
    \renewcommand{\subsectionmark}[1]{\markright{#1}{}}
}
