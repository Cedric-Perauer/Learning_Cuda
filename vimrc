set nocompatible              " be iMproved, required
filetype off                  " required
set completeopt+=noinsert 

let g:airline_theme='simple'
let g:rainbow_active = 1
" set the runtime path to include Vundle and initialize
map <silent> <C-a> :NERDTreeToggle<CR>
map <silent> <C-n>  :RainbowLevelsToggle<CR>

hi! RainbowLevel0 ctermbg=240 guibg=#585858
hi! RainbowLevel1 ctermbg=239 guibg=#4e4e4e
hi! RainbowLevel2 ctermbg=238 guibg=#444444
hi! RainbowLevel3 ctermbg=237 guibg=#3a3a3a
hi! RainbowLevel4 ctermbg=236 guibg=#303030
hi! RainbowLevel5 ctermbg=235 guibg=#262626
hi! RainbowLevel6 ctermbg=234 guibg=#1c1c1c
hi! RainbowLevel7 ctermbg=233 guibg=#121212
hi! RainbowLevel8 ctermbg=232 guibg=#080808
autocmd vimenter * NERDTree
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
Plugin 'frazrepo/vim-rainbow'
Plugin 'vim-airline/vim-airline'
Plugin 'vim-airline/vim-airline-themes'
Plugin 'zxqfl/tabnine-vim'
Plugin 'thiagoalessio/rainbow_levels.vim'
Plugin 'preservim/nerdtree'
" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo
Plugin 'tpope/vim-fugitive'
" plugin from http://vim-scripts.org/vim/scripts.html
" Plugin 'L9'
" Git plugin not hosted on GitHub
Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
Plugin 'file:///home/gmarik/path/to/plugin'
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" Install L9 and avoid a Naming conflict if you've already installed a
" different version somewhere else.
" Plugin 'ascenator/L9', {'name': 'newL9'}

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line
