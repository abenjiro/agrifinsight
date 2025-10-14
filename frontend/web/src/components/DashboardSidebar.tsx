import { Link, useLocation } from 'react-router-dom'
import {
  Home,
  BarChart3,
  Camera,
  Lightbulb,
  Settings,
  HelpCircle,
  X
} from 'lucide-react'
import { useSidebar } from '../contexts/SidebarContext'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: Home, gradient: 'from-purple-700 to-pink-500' },
  { name: 'Analysis', href: '/analysis', icon: Camera, gradient: 'from-blue-600 to-cyan-400' },
  { name: 'Recommendations', href: '/recommendations', icon: Lightbulb, gradient: 'from-green-600 to-lime-400' },
  { name: 'Reports', href: '/reports', icon: BarChart3, gradient: 'from-red-600 to-rose-400' },
  { name: 'Settings', href: '/settings', icon: Settings, gradient: 'from-gray-600 to-gray-400' },
  { name: 'Help', href: '/help', icon: HelpCircle, gradient: 'from-yellow-600 to-yellow-400' },
]

export function DashboardSidebar() {
  const location = useLocation()
  const { isOpen, setIsOpen } = useSidebar()

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-[800] xl:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`max-w-62.5 w-62.5 ease-nav-brand z-[990] fixed inset-y-0 my-4 ml-4 block ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } flex-wrap items-center justify-between overflow-y-auto rounded-2xl border-0 bg-white p-0 antialiased shadow-xl transition-transform duration-200 xl:left-0 xl:translate-x-0 xl:bg-transparent`}
      >
        {/* Logo Section */}
        <div className="h-19.5">
          <button
            className="absolute top-0 right-0 p-4 opacity-50 cursor-pointer text-slate-400 xl:hidden"
            onClick={() => setIsOpen(false)}
          >
            <X className="w-5 h-5" />
          </button>
          <Link
            to="/"
            className="block px-8 py-6 m-0 text-sm whitespace-nowrap text-slate-700"
          >
            <div className="flex items-center">
              <div className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-green-600 to-lime-400 text-white font-bold">
                A
              </div>
              <span className="ml-3 font-semibold transition-all duration-200 ease-nav-brand">
                AgriFinSight
              </span>
            </div>
          </Link>
        </div>

        <hr className="h-px mt-0 bg-transparent bg-gradient-to-r from-transparent via-black/40 to-transparent" />

        {/* Navigation */}
        <div className="items-center block w-auto max-h-screen overflow-auto h-sidenav grow basis-full">
          <ul className="flex flex-col pl-0 mb-0">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              const Icon = item.icon

              return (
                <li key={item.name} className="mt-0.5 w-full">
                  <Link
                    to={item.href}
                    className={`py-3 text-sm ease-nav-brand my-0 mx-4 flex items-center whitespace-nowrap px-4 transition-colors ${
                      isActive
                        ? 'shadow-soft-xl rounded-lg bg-white font-semibold text-slate-700'
                        : 'text-slate-700'
                    }`}
                    onClick={() => setIsOpen(false)}
                  >
                    <div
                      className={`shadow-soft-2xl mr-2 flex h-8 w-8 items-center justify-center rounded-lg bg-center stroke-0 text-center xl:p-2.5 ${
                        isActive
                          ? `bg-gradient-to-tl ${item.gradient}`
                          : 'bg-white'
                      }`}
                    >
                      <Icon
                        className={`${
                          isActive ? 'text-white' : 'text-slate-800 opacity-60'
                        }`}
                        size={12}
                      />
                    </div>
                    <span className="ml-1 duration-300 opacity-100 pointer-events-none ease-soft">
                      {item.name}
                    </span>
                  </Link>
                </li>
              )
            })}
          </ul>
        </div>

        {/* Card at Bottom */}
        <div className="mx-4 my-4">
          <div className="relative flex flex-col min-w-0 break-words bg-gradient-to-tl from-purple-700 to-pink-500 rounded-2xl bg-clip-border shadow-soft-xl">
            <div className="flex-auto p-4">
              <div className="text-white">
                <h6 className="mb-0 text-white font-semibold">Need Help?</h6>
                <p className="mb-4 text-xs font-normal leading-tight text-white opacity-80">
                  Check our documentation
                </p>
                <button className="inline-block w-full px-6 py-2.5 mb-0 font-bold text-center uppercase align-middle transition-all bg-transparent border-0 rounded-lg cursor-pointer leading-normal text-sm ease-soft-in tracking-tight-soft shadow-soft-md bg-150 bg-x-25 hover:scale-102 active:opacity-85 text-white/80 hover:text-white">
                  Documentation
                </button>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}
