import { ReactNode } from 'react'
import { DashboardHeader } from './DashboardHeader'
import { DashboardSidebar } from './DashboardSidebar'
import { SidebarProvider } from '../contexts/SidebarContext'

interface LayoutProps {
  children: ReactNode
}

export function DashboardLayout({ children }: LayoutProps) {
  return (
    <SidebarProvider>
      <div className="m-0 font-sans text-base antialiased font-normal leading-default bg-gray-50 text-slate-500 overflow-x-hidden">
        {/* Sidebar */}
        <DashboardSidebar />

        {/* Main Content Area */}
        <main className="ease-soft-in-out relative h-full max-h-screen overflow-y-auto overflow-x-hidden transition-all duration-200 xl:ml-[17.5rem]">
          {/* Top Navbar */}
          <DashboardHeader />

          {/* Page Content Container */}
          <div className="w-full px-6 py-6 mx-auto">
            <div className="flex flex-wrap -mx-3">
              {children}
            </div>
          </div>
        </main>
      </div>
    </SidebarProvider>
  )
}
