import React from 'react';
import classNames from 'classnames';

interface BaseDashboardCardProps {
  children: React.ReactNode;
  title?: string;
  badge?: React.ReactNode;
  className?: string;
  hover?: boolean;
  noPadding?: boolean;
}

export function BaseDashboardCard({ 
  children, 
  title, 
  badge, 
  className = '', 
  hover = false,
  noPadding = false 
}: BaseDashboardCardProps) {
  return (
    <div 
      className={classNames(
        'bg-white rounded-lg shadow-lg border border-gray-100',
        {
          'hover:shadow-xl transition-shadow duration-200': hover,
          'p-6': !noPadding,
          'mb-6': !className.includes('mb-'),
          'mb-8': title && !className.includes('mb-'), // More space for cards with titles
        },
        className
      )}
    >
      {title && (
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">
            {title}
          </h2>
          {badge && badge}
        </div>
      )}
      {children}
    </div>
  );
}

// Convenience components for common card patterns
export function DashboardCardHeader({ 
  title, 
  badge 
}: { 
  title: string; 
  badge?: React.ReactNode; 
}) {
  return (
    <div className="flex items-center justify-between mb-4">
      <h2 className="text-xl font-semibold text-gray-900">
        {title}
      </h2>
      {badge && badge}
    </div>
  );
}

export function DashboardCardContent({ 
  children, 
  className = '' 
}: { 
  children: React.ReactNode; 
  className?: string; 
}) {
  return (
    <div className={classNames('space-y-3', className)}>
      {children}
    </div>
  );
}