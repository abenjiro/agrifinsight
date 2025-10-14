import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Search, User, Settings, LogOut, Bell } from 'lucide-react'
import { useSidebar } from '../contexts/SidebarContext'

export function DashboardHeader() {
  const [isProfileOpen, setIsProfileOpen] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const { toggleSidebar } = useSidebar()

  const handleLogout = () => {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('refresh_token')
    localStorage.removeItem('user')
    navigate('/login')
  }

  // Get page title from pathname
  const getPageTitle = () => {
    const path = location.pathname
    if (path.includes('dashboard')) return 'Dashboard'
    if (path.includes('analysis')) return 'Analysis'
    if (path.includes('recommendations')) return 'Recommendations'
    if (path.includes('reports')) return 'Reports'
    if (path.includes('settings')) return 'Settings'
    if (path.includes('help')) return 'Help'
    return 'Dashboard'
  }

  // Get user info from localStorage
  const getUserInfo = () => {
    const userStr = localStorage.getItem('user')
    if (userStr) {
      try {
        const user = JSON.parse(userStr)
        return { name: user.email?.split('@')[0] || 'User', email: user.email }
      } catch {
        return { name: 'User', email: '' }
      }
    }
    return { name: 'User', email: '' }
  }

  const user = getUserInfo()

  return (
    <nav className="sticky top-0 z-50 flex flex-wrap items-center justify-between px-0 py-2 mx-2 sm:mx-6 transition-all shadow-none duration-250 ease-soft-in rounded-2xl lg:flex-nowrap lg:justify-start backdrop-blur-md bg-white/80">
      <div className="flex items-center justify-between w-full px-4 py-1 mx-auto flex-wrap-inherit">
        {/* Breadcrumb */}
        <nav className="hidden sm:block">
          <ol className="flex flex-wrap pt-1 mr-12 bg-transparent rounded-lg sm:mr-16">
            <li className="leading-normal text-sm">
              <a className="opacity-50 text-slate-700" href="/dashboard">
                Pages
              </a>
            </li>
            <li className="text-sm pl-2 capitalize leading-normal text-slate-700 before:float-left before:pr-2 before:text-gray-600 before:content-['/']">
              {getPageTitle()}
            </li>
          </ol>
          <h6 className="mb-0 font-bold capitalize">{getPageTitle()}</h6>
        </nav>

        <div className="flex items-center mt-2 grow sm:mt-0 sm:mr-6 md:mr-0 lg:flex lg:basis-auto">
          {/* Search Bar */}
          <div className="flex items-center md:ml-auto md:pr-4">
            <div className="relative flex flex-wrap items-stretch w-full transition-all rounded-lg ease-soft">
              <span className="text-sm ease-soft leading-5.6 absolute z-50 -ml-px flex h-full items-center whitespace-nowrap rounded-lg rounded-tr-none rounded-br-none border border-r-0 border-transparent bg-transparent py-2 px-2.5 text-center font-normal text-slate-500 transition-all">
                <Search className="w-4 h-4" />
              </span>
              <input
                type="text"
                className="pl-9 text-sm focus:shadow-soft-primary-outline ease-soft w-1/100 leading-5.6 relative -ml-px block min-w-0 flex-auto rounded-lg border border-solid border-gray-300 bg-white bg-clip-padding py-2 pr-3 text-gray-700 transition-all placeholder:text-gray-500 focus:border-fuchsia-300 focus:outline-none focus:transition-shadow"
                placeholder="Type here..."
              />
            </div>
          </div>

          {/* Right Side Icons */}
          <ul className="flex flex-row justify-end pl-0 mb-0 list-none md-max:w-full">
            {/* Notifications */}
            <li className="flex items-center px-4">
              <button className="p-0 transition-all text-sm ease-nav-brand text-slate-500 hover:text-slate-700">
                <Bell className="w-5 h-5" />
              </button>
            </li>

            {/* User Profile Dropdown */}
            <li className="flex items-center relative">
              <button
                onClick={() => setIsProfileOpen(!isProfileOpen)}
                className="flex items-center px-0 py-2 font-semibold transition-all ease-nav-brand text-sm text-slate-500 hover:text-slate-700"
              >
                <User className="w-5 h-5 sm:mr-1" />
                <span className="hidden sm:inline capitalize">{user.name}</span>
              </button>

              {/* Dropdown Menu */}
              {isProfileOpen && (
                <>
                  {/* Overlay */}
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setIsProfileOpen(false)}
                  />

                  {/* Dropdown */}
                  <div className="absolute right-0 top-full mt-2 w-56 bg-white rounded-lg shadow-soft-xl py-2 z-50">
                    <div className="px-4 py-3 border-b border-gray-100">
                      <p className="text-sm font-semibold text-gray-900 capitalize">{user.name}</p>
                      <p className="text-xs text-gray-500 truncate">{user.email}</p>
                    </div>

                    <button
                      onClick={() => {
                        setIsProfileOpen(false)
                        navigate('/settings')
                      }}
                      className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <Settings className="w-4 h-4 mr-3" />
                      Settings
                    </button>

                    <button
                      onClick={handleLogout}
                      className="flex items-center w-full px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                    >
                      <LogOut className="w-4 h-4 mr-3" />
                      Logout
                    </button>
                  </div>
                </>
              )}
            </li>

            {/* Mobile Sidebar Toggle */}
            <li className="flex items-center pl-4 xl:hidden">
              <button
                onClick={toggleSidebar}
                className="block p-2 transition-all ease-nav-brand text-sm text-slate-500 hover:text-slate-700"
                id="iconNavbarSidenav"
              >
                <div className="w-5 h-4 flex flex-col justify-between">
                  <span className="ease-soft relative block h-0.5 w-full rounded-sm bg-slate-500 transition-all"></span>
                  <span className="ease-soft relative block h-0.5 w-full rounded-sm bg-slate-500 transition-all"></span>
                  <span className="ease-soft relative block h-0.5 w-full rounded-sm bg-slate-500 transition-all"></span>
                </div>
              </button>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  )
}
