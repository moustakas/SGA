"""
This holds custom filters for the Sample model, 
which work by selecting Sample objects in the database based on meeting the desired criteria.
"""
import django_filters
from SGA.webapp.sample.models import Sample

class SampleFilter(django_filters.FilterSet):
    """
    Our custom filter for the centrals model.
    Filter options include greater than or equal to, and less than or equal to 
    on the following Centrals fields: ra, dec, z, and la.
    The filter can be used in a form on our webpage, as seen in list.html. 
    """
    #field_name is the Centrals object variable
    #lookup_expr is used to get ranges (currently using greater/less than or equal to  
    ra__gte = django_filters.NumberFilter(field_name='ra', lookup_expr='gte')
    ra__lte = django_filters.NumberFilter(field_name='ra', lookup_expr='lte')

    dec__gte = django_filters.NumberFilter(field_name='dec', lookup_expr='gte')
    dec__lte = django_filters.NumberFilter(field_name='dec', lookup_expr='lte')

    sgaid__gte = django_filters.NumberFilter(field_name='sga_id', lookup_expr='gte')
    sgaid__lte = django_filters.NumberFilter(field_name='sga_id', lookup_expr='lte')

    class Meta:
        model = Sample
        #add variable to fields[] if looking for exact match
        fields = []

        def id(self):
            return self.sga_id

