; Group LSLGA galaxies together for any where their circular radii
; would overlap.  Use the catalog D25 diameters (in arcmin) multiplied
; by a scaling factor MFAC (default value 1.25).  The output catalog
; adds the column GROUPNUM that is -1 for any galaxies outside the
; footprint (IN_DESI='F'), otherwise has a unique value for each group.
; The column MULTGROUP is the multiplicity of that galaxy's group.
pro lslga_group,infile,outfile,mfac=mfac

if (NOT keyword_set(infile)) then infile='LSLGA-v5.0.fits'
if (NOT keyword_set(outfile)) then outfile='LSLGA-v5.0-group.fits'
if (NOT keyword_set(mfac)) then mfac=1.25

; Read catalog, trim to DESI DR9 footprint
cat_in=mrdfits(infile,1)
ii=where(cat_in.in_desi EQ 'T')
cat=cat_in[ii]

; Initialize a unique group number for each galaxy
gnum=lindgen(n_elements(cat))
mgrp=lonarr(n_elements(cat))+1L

; First group galaxies within 10 arcmin, setting those to have
; the same group number
dmax=10./60
ingroup=spheregroup(cat.ra,cat.dec,dmax,multgroup=multgroup,$
 firstgroup=firstgroup,nextgroup=nextgroup)
ngroup=n_elements(firstgroup NE -1)
for i=0L,ngroup-1L do begin
   print,i,ngroup
   nn=multgroup[i] ; number of galaxies in this group
   if (nn GT 1) then begin
      ; Build INDX as the indices of all objects in this grouping
      indx=lonarr(nn)
      indx[0]=firstgroup[i]
      for j=0L,nn-2L do indx[j+1]=nextgroup[indx[j]]
      ; Look at all pairs within this grouping to see if they
      ; should be connected
      for j=0L, nn-2L do begin
         for k=j, nn-1L do begin
            dd = djs_diff_angle(cat[indx[j]].ra,cat[indx[j]].dec,$
             cat[indx[k]].ra,cat[indx[k]].dec)
            if (dd LT 0.5*mfac*(cat[indx[j]].d25/60.+cat[indx[k]].d25/60.)) then begin
               ; Found that these two galaxies should be connected,
               ; so make GNUM the same for them...
               jndx = where(gnum[indx] EQ gnum[indx[j]] OR gnum[indx] EQ gnum[indx[k]],ct)
               gnum[indx[jndx]] = gnum[indx[jndx[0]]]
               mgrp[indx[jndx]] = ct
            endif
         endfor
      endfor
   endif
endfor

; Special-case the largest galaxies, looking for neighbhors
ibig=where(cat.d25/60. GT dmax,nbig)
for i=0L,nbig-1L do begin
   print,i,nbig
   dd = djs_diff_angle(cat[ibig[i]].ra,cat[ibig[i]].dec,cat.ra,cat.dec)
   inear = where(dd LT 0.5*(cat[ibig[i]].d25+cat.d25)/60.,nnear)
   for j=0L,nnear-1L do begin
      indx=where(gnum EQ gnum[ibig[i]] OR gnum EQ gnum[inear[j]],ct)
      gnum[indx]=gnum[indx[0]]
      mgrp[indx]=ct
   endfor
endfor

; Add a column GROUPNUM to the catalog
addstr=replicate({groupnum:-1L,multgroup:0L},n_elements(cat_in))
cat_out=struct_addtags(cat_in,addstr)
cat_out[ii].groupnum=gnum
cat_out[ii].multgroup=mgrp
mwrfits,cat_out,outfile,/create

end

